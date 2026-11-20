"""Internal API for local FunASR and explicitly selected Qwen transcription."""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
import subprocess
import tempfile
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

MEDIA_ROOT = Path(os.getenv("DTK_TRANSCRIBER_MEDIA_ROOT", "/var/lib/dtk/media"))
MODEL_PATH = Path(os.getenv("DTK_TRANSCRIBER_MODEL_DIR", "/models/funasr"))
TOKEN = os.getenv("DTK_TRANSCRIBER_TOKEN", "")
WORKERS = max(1, int(os.getenv("DTK_TRANSCRIBER_WORKERS", "1")))
QUEUE_SIZE = max(1, int(os.getenv("DTK_TRANSCRIBER_QUEUE", "8")))
QWEN_WORKERS = max(1, int(os.getenv("DTK_TRANSCRIBER_QWEN_WORKERS", "2")))
QWEN_QUEUE_SIZE = max(1, int(os.getenv("DTK_TRANSCRIBER_QWEN_QUEUE_SIZE", "8")))
QWEN_API_KEY = os.getenv("DTK_TRANSCRIBER_QWEN_API_KEY", "")
QWEN_BASE_URL = os.getenv("DTK_TRANSCRIBER_QWEN_BASE_URL", "").rstrip("/")
QWEN_TIMEOUT_SECONDS = max(1.0, float(os.getenv("DTK_TRANSCRIBER_QWEN_TIMEOUT_SECONDS", "300")))
QWEN_MAX_RETRIES = max(0, int(os.getenv("DTK_TRANSCRIBER_QWEN_MAX_RETRIES", "2")))
QWEN_CHUNK_SECONDS = 270
QWEN_MAX_DATA_BYTES = 10 * 1024 * 1024
JOB_HISTORY_SIZE = max(
    WORKERS + QUEUE_SIZE + QWEN_WORKERS + QWEN_QUEUE_SIZE,
    1,
    int(os.getenv("DTK_TRANSCRIBER_JOB_HISTORY", "64")),
)
THREADS = max(1, int(os.getenv("DTK_TRANSCRIBER_THREADS", "2")))
PROGRESS_LOG_SECONDS = max(5.0, float(os.getenv("DTK_TRANSCRIBER_PROGRESS_LOG_SECONDS", "30")))
#: The worker gives up on a job after this long, so a decode still running past
#: it has no caller left. It is also what stops one file ffmpeg cannot finish
#: from holding a worker until the container restarts.
DECODE_TIMEOUT_SECONDS = max(60, int(os.getenv("DTK_TRANSCRIBER_TIMEOUT_SECONDS", "3600")))
MODEL_NAME = "paraformer-zh-large-onnx-int8"
QWEN_MODEL_NAME = "qwen3-asr-flash"
NORMALIZER_NAME = "wetext-zh-itn-0.1.8"
VALID_FAILURE_POLICIES = frozenset({"fallback", "fail"})
_policy_value = os.getenv("DTK_TRANSCRIBER_NORMALIZATION_FAILURE_POLICY", "fallback").lower()
POLICY_ERROR = (
    "DTK_TRANSCRIBER_NORMALIZATION_FAILURE_POLICY must be fallback or fail"
    if _policy_value not in VALID_FAILURE_POLICIES
    else ""
)
DEFAULT_FAILURE_POLICY: Literal["fallback", "fail"] = (
    "fail" if _policy_value == "fail" else "fallback"
)
log = logging.getLogger("uvicorn.error.transcriber")
log.setLevel(logging.INFO)


def _safe_error(exc: BaseException) -> str:
    message = str(exc).replace(str(MEDIA_ROOT), "<media-root>")
    return message.replace(str(MODEL_PATH), "<model-root>")[:300]


ProgressCallback = Callable[[float, float], None]
StageName = Literal["preparing_audio", "recognizing", "post_processing"]
StepName = Literal[
    "decoding_audio",
    "extracting_and_splitting_audio",
    "detecting_speech",
    "local_inference",
    "external_request",
    "merging_results",
    "adding_punctuation",
    "normalizing_text",
]
TransitionCallback = Callable[[StageName, StepName], None]
SegmentCallback = Callable[[str], None]
ChunkCallback = Callable[[int, int], None]
VALID_STEPS_BY_STAGE: dict[StageName, frozenset[StepName]] = {
    "preparing_audio": frozenset({"decoding_audio", "extracting_and_splitting_audio"}),
    "recognizing": frozenset({"detecting_speech", "local_inference", "external_request"}),
    "post_processing": frozenset({"merging_results", "adding_punctuation", "normalizing_text"}),
}


class NormalizationFailed(RuntimeError):
    pass


class JobCancelled(Exception):
    """Raised from a progress callback to stop a backend whose job was cancelled."""


class Backend(Protocol):
    def transcribe(
        self,
        path: Path,
        normalize: bool,
        normalization_failure_policy: str,
        progress: ProgressCallback | None = None,
        transition: TransitionCallback | None = None,
        segment: SegmentCallback | None = None,
    ) -> dict[str, Any]: ...


class QwenRequestFailed(RuntimeError):
    pass


def _normalizer() -> Any:
    from wetext import Normalizer

    return Normalizer(
        lang="zh",
        operator="itn",
        enable_0_to_9=False,
        traditional_to_simple=True,
        full_to_half=False,
    )


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "preds", "value"):
            if key in value:
                return _first_text(value[key])
    if isinstance(value, list | tuple) and value:
        return _first_text(value[0])
    return ""


def _segments(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        for key in ("segments", "value", "timestamps"):
            if key in value:
                return _segments(value[key])
        return []
    if not isinstance(value, list | tuple) or not value:
        return []
    if len(value) == 1 and isinstance(value[0], list | tuple | dict):
        nested = _segments(value[0])
        if nested:
            return nested
    found: list[tuple[int, int]] = []
    for item in value:
        if isinstance(item, list | tuple) and len(item) >= 2:
            try:
                start, end = int(item[0]), int(item[1])
            except (TypeError, ValueError):
                continue
            if 0 <= start < end:
                found.append((start, end))
    return found


class FunASRBackend:
    def __init__(self) -> None:
        from funasr_onnx import CT_Transformer, Fsmn_vad, Paraformer
        from imageio_ffmpeg import get_ffmpeg_exe

        # Resolve and execute `-version` during backend startup. A wheel missing
        # its platform binary therefore degrades health immediately instead of
        # failing only after a user submits a transcription.
        self._ffmpeg_exe = get_ffmpeg_exe()
        common = {"device_id": -1, "quantize": True, "intra_op_num_threads": THREADS}
        self._asr = Paraformer(str(MODEL_PATH / "asr"), batch_size=1, **common)
        self._vad = Fsmn_vad(str(MODEL_PATH / "vad"), batch_size=1, **common)
        self._punc = CT_Transformer(str(MODEL_PATH / "punc"), **common)
        self._normalizer = _normalizer()

    def _decode(self, path: Path) -> Any:
        import numpy as np

        try:
            completed = subprocess.run(
                [
                    self._ffmpeg_exe,
                    "-v",
                    "error",
                    "-nostdin",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    "-",
                ],
                check=False,
                capture_output=True,
                timeout=DECODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"ffmpeg timed out after {DECODE_TIMEOUT_SECONDS}s decoding the audio track"
            ) from None
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or "ffmpeg could not decode the first audio track")
        return np.frombuffer(completed.stdout, dtype=np.float32).copy()

    def transcribe(
        self,
        path: Path,
        normalize: bool,
        normalization_failure_policy: str,
        progress: ProgressCallback | None = None,
        transition: TransitionCallback | None = None,
        segment: SegmentCallback | None = None,
    ) -> dict[str, Any]:
        if transition:
            transition("preparing_audio", "decoding_audio")
        audio = self._decode(path)
        duration = len(audio) / 16000.0
        if progress:
            progress(0.0, duration)
        if not len(audio):
            return self._result("", duration, normalize, False, None)

        if transition:
            transition("recognizing", "detecting_speech")
        intervals = _segments(self._vad(audio))
        if not intervals:
            return self._result("", duration, normalize, False, None)

        if transition:
            transition("recognizing", "local_inference")
        parts: list[str] = []
        for start_ms, end_ms in intervals:
            start = max(0, round(start_ms * 16))
            end = min(len(audio), round(end_ms * 16))
            if end <= start:
                continue
            text = _first_text(self._asr(audio[start:end])).strip()
            if text:
                parts.append(text)
                if segment:
                    # This is deliberately the raw, append-only ASR fragment.
                    # Punctuation and ITN run over the whole transcript later,
                    # so callers must treat it as provisional until the job is
                    # done.
                    segment(text)
            if progress:
                progress(min(duration, end_ms / 1000.0), duration)

        raw_text = "".join(parts).strip()
        if raw_text and transition:
            transition("post_processing", "adding_punctuation")
        punctuated = (_first_text(self._punc(raw_text)).strip() or raw_text) if raw_text else ""

        if not normalize or not punctuated:
            return self._result(punctuated, duration, normalize, False, None)
        if transition:
            transition("post_processing", "normalizing_text")
        try:
            normalized = str(self._normalizer.normalize(punctuated)).strip()
        except Exception as exc:
            if normalization_failure_policy == "fail":
                raise NormalizationFailed("NORMALIZATION_FAILED") from exc
            return self._result(punctuated, duration, True, False, "NORMALIZATION_FAILED")
        return self._result(normalized, duration, True, True, None)

    @staticmethod
    def _result(
        text: str,
        duration: float,
        requested: bool,
        applied: bool,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "duration_ms": round(duration * 1000),
            "normalization_requested": requested,
            "normalization_applied": applied,
            "normalizer": NORMALIZER_NAME if requested else None,
            "normalization_error": error,
        }


class QwenBackend:
    """Qwen3-ASR-Flash adapter using bounded Base64 audio chunks."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        if not QWEN_API_KEY or not QWEN_BASE_URL:
            raise RuntimeError("Qwen API key and base URL are not configured")
        from imageio_ffmpeg import get_ffmpeg_exe

        self._ffmpeg_exe = get_ffmpeg_exe()
        self._normalizer = _normalizer()
        self._client = client or httpx.AsyncClient(timeout=QWEN_TIMEOUT_SECONDS)
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _duration(progress: str) -> float:
        values = [line[9:] for line in progress.splitlines() if line.startswith("out_time=")]
        if not values:
            return 0.0
        try:
            hours, minutes, seconds = values[-1].split(":", 2)
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (TypeError, ValueError):
            return 0.0

    async def _extract(self, path: Path, destination: Path) -> tuple[list[Path], float]:
        pattern = destination / "chunk-%04d.mp3"
        process = await asyncio.create_subprocess_exec(
            self._ffmpeg_exe,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-f",
            "segment",
            "-segment_time",
            str(QWEN_CHUNK_SECONDS),
            "-reset_timestamps",
            "1",
            "-progress",
            "pipe:1",
            str(pattern),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=DECODE_TIMEOUT_SECONDS
            )
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise RuntimeError(
                f"ffmpeg timed out after {DECODE_TIMEOUT_SECONDS}s extracting audio"
            ) from None
        except asyncio.CancelledError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or "ffmpeg could not extract audio")
        chunks = sorted(destination.glob("chunk-*.mp3"))
        if not chunks:
            raise RuntimeError("the media contains no decodable audio")
        return chunks, self._duration(stdout.decode("utf-8", errors="replace"))

    @staticmethod
    def _retry_after(response: httpx.Response) -> float:
        try:
            return min(30.0, max(0.0, float(response.headers.get("retry-after", "0"))))
        except ValueError:
            return 0.0

    async def _recognize(self, chunk: Path, *, job_id: str, index: int) -> str:
        raw = await asyncio.to_thread(chunk.read_bytes)
        encoded = base64.b64encode(raw)
        if len(raw) > QWEN_MAX_DATA_BYTES or len(encoded) > QWEN_MAX_DATA_BYTES:
            raise QwenRequestFailed("Qwen audio chunk exceeds the 10 MB input limit")
        payload = {
            "model": QWEN_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:audio/mpeg;base64,{encoded.decode('ascii')}"
                            },
                        }
                    ],
                }
            ],
            "stream": False,
            "asr_options": {"enable_itn": False},
        }
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(QWEN_MAX_RETRIES + 1):
            began = time.monotonic()
            try:
                response = await self._client.post(
                    f"{QWEN_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                    json=payload,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt >= QWEN_MAX_RETRIES:
                    raise QwenRequestFailed(f"Qwen request failed: {type(exc).__name__}") from exc
                await asyncio.sleep(min(30.0, 2**attempt))
                continue
            request_id = response.headers.get("x-request-id") or response.headers.get(
                "x-dashscope-request-id", "unknown"
            )
            log.info(
                "transcriber.qwen.response job_id=%s chunk=%d attempt=%d status=%d "
                "elapsed_ms=%d request_id=%s",
                job_id,
                index,
                attempt + 1,
                response.status_code,
                round((time.monotonic() - began) * 1000),
                _loggable(request_id),
            )
            if response.status_code in retryable and attempt < QWEN_MAX_RETRIES:
                await asyncio.sleep(self._retry_after(response) or min(30.0, 2**attempt))
                continue
            if response.status_code >= 400:
                raise QwenRequestFailed(f"Qwen API returned HTTP {response.status_code}")
            try:
                body = response.json()
                text = body["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise QwenRequestFailed("Qwen API returned an invalid response") from exc
            if not isinstance(text, str):
                raise QwenRequestFailed("Qwen API returned an invalid transcript")
            return text.strip()
        raise QwenRequestFailed("Qwen request failed")

    async def transcribe(
        self,
        path: Path,
        normalize: bool,
        normalization_failure_policy: str,
        *,
        job_id: str,
        progress: ProgressCallback | None = None,
        transition: TransitionCallback | None = None,
        segment: SegmentCallback | None = None,
        chunk: ChunkCallback | None = None,
    ) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix=f"dtk-transcriber-{job_id}-") as temporary:
            if transition:
                transition("preparing_audio", "extracting_and_splitting_audio")
            chunks, duration = await self._extract(path, Path(temporary))
            if progress:
                progress(0.0, duration)
            if transition:
                transition("recognizing", "external_request")
            parts: list[str] = []
            for index, audio in enumerate(chunks, start=1):
                if chunk:
                    chunk(index, len(chunks))
                text = await self._recognize(audio, job_id=job_id, index=index)
                if text:
                    parts.append(text)
                    if segment:
                        segment(text)
                if progress:
                    progress(min(duration, index * QWEN_CHUNK_SECONDS), duration)

        if parts and transition:
            transition("post_processing", "merging_results")
        transcript = "\n".join(parts).strip()
        if not normalize or not transcript:
            return FunASRBackend._result(transcript, duration, normalize, False, None)
        if transition:
            transition("post_processing", "normalizing_text")
        try:
            normalized = str(self._normalizer.normalize(transcript)).strip()
        except Exception as exc:
            if normalization_failure_policy == "fail":
                raise NormalizationFailed("NORMALIZATION_FAILED") from exc
            return FunASRBackend._result(transcript, duration, True, False, "NORMALIZATION_FAILED")
        return FunASRBackend._result(normalized, duration, True, True, None)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    download_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    path: str = Field(min_length=1, max_length=1024)
    engine: Literal["local", "qwen"] = "local"
    normalize: bool = True
    normalization_failure_policy: Literal["fallback", "fail"] = DEFAULT_FAILURE_POLICY
    model: Literal["paraformer-zh-large-onnx-int8", "qwen3-asr-flash"] = MODEL_NAME

    @model_validator(mode="after")
    def matching_engine_and_model(self) -> JobRequest:
        expected = MODEL_NAME if self.engine == "local" else QWEN_MODEL_NAME
        if self.model != expected:
            raise ValueError(f"model must be {expected} for engine {self.engine}")
        return self


def _validate_models() -> None:
    for component in ("asr", "vad", "punc"):
        directory = MODEL_PATH / component
        for name in ("model_quant.onnx", "config.yaml"):
            if not (directory / name).is_file():
                raise RuntimeError(f"offline {component} model is missing {name}")


def _media_file(relative: str) -> Path:
    """Resolve a caller's path inside the media root, or refuse it.

    Normalize, then check the prefix. The first version resolved the path and
    called relative_to, which is just as safe, and CodeQL reported it as path
    injection on the PR that added this file: it recognizes this shape as a
    sanitizer and not that one. Symlinks are resolved before the check, so a
    link inside the root that points out of it is still refused.
    """
    if os.path.isabs(relative) or ".." in Path(relative).parts:
        raise HTTPException(status_code=400, detail="path must be relative to the media root")
    root = os.path.realpath(MEDIA_ROOT)
    candidate = os.path.realpath(os.path.join(root, relative))
    if not candidate.startswith(root + os.sep):
        raise HTTPException(status_code=400, detail="media path is unavailable or escapes root")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=400, detail="media path is not a file")
    return Path(candidate)


def _loggable(value: object) -> str:
    """One log line per event, whatever a caller put in a field.

    The ids logged here are pattern-checked by the request model, so this
    changes nothing for a valid request; it exists so that no field taken from
    a request can start a forged line, which is what CodeQL asks of log calls.
    """
    return str(value).replace("\r", " ").replace("\n", " ")


def create_app(backend: Backend | None = None, qwen_backend: QwenBackend | None = None) -> FastAPI:
    jobs: dict[str, dict[str, Any]] = {}
    terminal_jobs: OrderedDict[str, None] = OrderedDict()
    queues: dict[str, asyncio.Queue[tuple[JobRequest, Path, dict[str, Any]]]] = {
        "local": asyncio.Queue(maxsize=QUEUE_SIZE),
        "qwen": asyncio.Queue(maxsize=QWEN_QUEUE_SIZE),
    }
    state: dict[str, Any] = {
        "local_backend": backend,
        "local_detail": POLICY_ERROR,
        "qwen_backend": qwen_backend,
        "qwen_detail": POLICY_ERROR,
    }
    active_qwen: dict[str, asyncio.Task[dict[str, Any]]] = {}

    def forget_job(job_id: str, expected: dict[str, Any] | None = None) -> None:
        if expected is not None and jobs.get(job_id) is not expected:
            return
        jobs.pop(job_id, None)
        terminal_jobs.pop(job_id, None)

    def remember_terminal(job_id: str, job: dict[str, Any]) -> None:
        # A final result already contains the whole transcript. Keeping its
        # provisional copy doubles the largest field in every retained job.
        job["partial_text"] = ""
        terminal_jobs[job_id] = None
        terminal_jobs.move_to_end(job_id)
        while len(terminal_jobs) > JOB_HISTORY_SIZE:
            expired_id, _ = terminal_jobs.popitem(last=False)
            jobs.pop(expired_id, None)

    async def run_worker(engine: Literal["local", "qwen"]) -> None:
        queue = queues[engine]
        while True:
            request, path, job = await queue.get()
            started = time.monotonic()
            heartbeat: asyncio.Task[None] | None = None

            # The backend runs in a thread that asyncio cannot interrupt, so a
            # cancelled job used to keep its worker until the last segment was
            # recognised - and with one worker, every queued job waited for
            # audio nobody wanted any more. Each callback is a point where the
            # backend can be stopped instead.
            def stop_if_cancelled(target: dict[str, Any]) -> None:
                if target["state"] == "cancelled":
                    raise JobCancelled

            def record_progress(
                processed_seconds: float,
                duration_seconds: float,
                target: dict[str, Any] = job,
            ) -> None:
                stop_if_cancelled(target)
                processed_ms = max(0, round(processed_seconds * 1000))
                duration_ms = max(0, round(duration_seconds * 1000))
                target["processed_ms"] = (
                    min(processed_ms, duration_ms) if duration_ms else processed_ms
                )
                target["media_duration_ms"] = duration_ms or None
                target["progress_percent"] = (
                    round(min(99.9, processed_ms * 100 / duration_ms), 1) if duration_ms else None
                )
                target["progress_revision"] += 1

            def record_transition(
                stage: StageName,
                step: StepName,
                target: dict[str, Any] = job,
                job_id: str = request.id,
                download_id: str | None = request.download_id,
            ) -> None:
                stop_if_cancelled(target)
                if step not in VALID_STEPS_BY_STAGE[stage]:
                    raise RuntimeError(f"invalid transcription transition: {stage}/{step}")
                if target.get("stage") == stage and target.get("step") == step:
                    return
                transition_update: dict[str, Any] = {
                    "stage": stage,
                    "step": step,
                    "progress_revision": target["progress_revision"] + 1,
                }
                if stage == "post_processing":
                    transition_update["progress_percent"] = 99.9
                target.update(transition_update)
                log.info(
                    "transcriber.job.stage job_id=%s download_id=%s stage=%s step=%s",
                    job_id,
                    download_id or "unknown",
                    stage,
                    step,
                )

            def record_segment(text: str, target: dict[str, Any] = job) -> None:
                stop_if_cancelled(target)
                if not text:
                    return
                target["partial_text"] = f"{target.get('partial_text') or ''}{text}"
                target["progress_revision"] += 1

            def record_chunk(index: int, count: int, target: dict[str, Any] = job) -> None:
                stop_if_cancelled(target)
                target["chunk_index"] = index
                target["chunk_count"] = count
                target["progress_revision"] += 1

            async def report_progress(
                target: dict[str, Any] = job,
                job_id: str = request.id,
                download_id: str | None = request.download_id,
                began: float = started,
            ) -> None:
                while True:
                    await asyncio.sleep(PROGRESS_LOG_SECONDS)
                    if target["state"] != "running":
                        return
                    log.info(
                        "transcriber.job.progress job_id=%s download_id=%s stage=%s step=%s "
                        "elapsed_ms=%d processed_ms=%s duration_ms=%s progress_percent=%s",
                        job_id,
                        download_id or "unknown",
                        target.get("stage") or "starting",
                        target.get("step") or "starting",
                        round((time.monotonic() - began) * 1000),
                        target.get("processed_ms") or 0,
                        target.get("media_duration_ms") or "unknown",
                        (
                            target.get("progress_percent")
                            if target.get("progress_percent") is not None
                            else "unknown"
                        ),
                    )

            try:
                if job["state"] == "cancelled":
                    continue
                job["state"] = "running"
                job["started_at"] = datetime.now(UTC).isoformat()
                log.info(
                    "transcriber.job.started job_id=%s download_id=%s engine=%s normalize=%s "
                    "normalization_failure_policy=%s",
                    request.id,
                    request.download_id or "unknown",
                    engine,
                    request.normalize,
                    request.normalization_failure_policy,
                )
                heartbeat = asyncio.create_task(report_progress())
                initial_step: StepName = (
                    "decoding_audio" if engine == "local" else "extracting_and_splitting_audio"
                )
                record_transition("preparing_audio", initial_step)
                if engine == "local":
                    result = await asyncio.to_thread(
                        state["local_backend"].transcribe,
                        path,
                        request.normalize,
                        request.normalization_failure_policy,
                        record_progress,
                        record_transition,
                        record_segment,
                    )
                else:
                    execution = asyncio.create_task(
                        state["qwen_backend"].transcribe(
                            path,
                            request.normalize,
                            request.normalization_failure_policy,
                            job_id=request.id,
                            progress=record_progress,
                            transition=record_transition,
                            segment=record_segment,
                            chunk=record_chunk,
                        )
                    )
                    active_qwen[request.id] = execution
                    try:
                        result = await execution
                    finally:
                        active_qwen.pop(request.id, None)
                if job["state"] != "cancelled":
                    job.update(result)
                    job.update(
                        {
                            "state": "done",
                            "stage": None,
                            "step": None,
                            "finished_at": datetime.now(UTC).isoformat(),
                            "processed_ms": result.get("duration_ms"),
                            "media_duration_ms": result.get("duration_ms"),
                            "progress_percent": 100.0,
                            "progress_revision": job["progress_revision"] + 1,
                        }
                    )
                    remember_terminal(request.id, job)
                    if result.get("normalization_error"):
                        log.warning(
                            "transcriber.job.normalization_fallback job_id=%s download_id=%s",
                            request.id,
                            request.download_id or "unknown",
                        )
                    log.info(
                        "transcriber.job.done job_id=%s download_id=%s elapsed_ms=%d "
                        "duration_ms=%s normalization_applied=%s",
                        request.id,
                        request.download_id or "unknown",
                        round((time.monotonic() - started) * 1000),
                        result.get("duration_ms"),
                        result.get("normalization_applied"),
                    )
            except asyncio.CancelledError:
                if job["state"] != "cancelled":
                    raise
            except Exception as exc:
                if job["state"] != "cancelled":
                    failed_stage = job.get("stage") or "unknown"
                    failed_step = job.get("step") or "unknown"
                    # Returned to API callers, so it gets the same scrubbing as the
                    # log line: ffmpeg names the file it could not read.
                    job.update(
                        {
                            "state": "failed",
                            "stage": None,
                            "step": None,
                            "finished_at": datetime.now(UTC).isoformat(),
                            "error": _safe_error(exc),
                            "progress_revision": job["progress_revision"] + 1,
                        }
                    )
                    if isinstance(exc, NormalizationFailed):
                        job["normalization_error"] = "NORMALIZATION_FAILED"
                    remember_terminal(request.id, job)
                    log.error(
                        "transcriber.job.failed job_id=%s download_id=%s stage=%s step=%s "
                        "elapsed_ms=%d error_type=%s detail=%s",
                        request.id,
                        request.download_id or "unknown",
                        failed_stage,
                        failed_step,
                        round((time.monotonic() - started) * 1000),
                        type(exc).__name__,
                        _safe_error(exc.__cause__ or exc),
                    )
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                    await asyncio.gather(heartbeat, return_exceptions=True)
                if job["state"] == "cancelled":
                    forget_job(request.id, job)
                queue.task_done()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["local_backend"] is None and not state["local_detail"]:
            try:
                _validate_models()
                log.info(
                    "transcriber.model.loading model=%s threads=%d quantized=true",
                    MODEL_NAME,
                    THREADS,
                )
                state["local_backend"] = await asyncio.to_thread(FunASRBackend)
                log.info("transcriber.model.ready model=%s", MODEL_NAME)
            except Exception as exc:
                state["local_detail"] = _safe_error(exc)
                log.error(
                    "transcriber.model.degraded error_type=%s detail=%s",
                    type(exc).__name__,
                    _safe_error(exc),
                )
        if state["qwen_backend"] is None and not state["qwen_detail"]:
            try:
                state["qwen_backend"] = QwenBackend()
                log.info("transcriber.qwen.ready model=%s", QWEN_MODEL_NAME)
            except Exception as exc:
                state["qwen_detail"] = _safe_error(exc)
                log.warning(
                    "transcriber.qwen.degraded error_type=%s detail=%s",
                    type(exc).__name__,
                    _safe_error(exc),
                )
        workers = [
            *(asyncio.create_task(run_worker("local")) for _ in range(WORKERS)),
            *(asyncio.create_task(run_worker("qwen")) for _ in range(QWEN_WORKERS)),
        ]
        log.info(
            "transcriber.service.ready local_ready=%s qwen_ready=%s local_workers=%d "
            "qwen_workers=%d local_queue_capacity=%d qwen_queue_capacity=%d "
            "normalization_failure_policy=%s",
            state["local_backend"] is not None and not state["local_detail"],
            state["qwen_backend"] is not None and not state["qwen_detail"],
            WORKERS,
            QWEN_WORKERS,
            QUEUE_SIZE,
            QWEN_QUEUE_SIZE,
            DEFAULT_FAILURE_POLICY,
        )
        try:
            yield
        finally:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            qwen = state.get("qwen_backend")
            if qwen is not None and hasattr(qwen, "aclose"):
                await qwen.aclose()

    app = FastAPI(title="DTK Transcriber", docs_url=None, redoc_url=None, lifespan=lifespan)

    def authorize(x_transcriber_token: str | None = Header(default=None)) -> None:
        supplied = (x_transcriber_token or "").encode()
        if TOKEN and not hmac.compare_digest(supplied, TOKEN.encode()):
            raise HTTPException(status_code=401, detail="invalid transcriber token")

    @app.get("/health")
    async def health(_: None = Depends(authorize)) -> dict[str, Any]:
        local_ready = state["local_backend"] is not None and not state["local_detail"]
        qwen_ready = state["qwen_backend"] is not None and not state["qwen_detail"]
        return {
            "status": "ok" if local_ready or qwen_ready else "degraded",
            "model": MODEL_NAME,
            "normalizer": NORMALIZER_NAME,
            "normalization_failure_policy": DEFAULT_FAILURE_POLICY,
            "workers": WORKERS,
            "queued": queues["local"].qsize() + queues["qwen"].qsize(),
            "running": sum(job["state"] == "running" for job in jobs.values()),
            "engines": {
                "local": {
                    "ready": local_ready,
                    "model": MODEL_NAME,
                    "workers": WORKERS,
                    "queued": queues["local"].qsize(),
                    "running": sum(
                        job["state"] == "running" and job["engine"] == "local"
                        for job in jobs.values()
                    ),
                    "detail": state["local_detail"],
                },
                "qwen": {
                    "ready": qwen_ready,
                    "model": QWEN_MODEL_NAME,
                    "workers": QWEN_WORKERS,
                    "queued": queues["qwen"].qsize(),
                    "running": sum(
                        job["state"] == "running" and job["engine"] == "qwen"
                        for job in jobs.values()
                    ),
                    "detail": state["qwen_detail"],
                },
            },
            "retained_terminal_jobs": len(terminal_jobs),
            "terminal_job_limit": JOB_HISTORY_SIZE,
            "detail": ""
            if local_ready or qwen_ready
            else (state["local_detail"] or state["qwen_detail"]),
        }

    @app.post("/jobs", status_code=202)
    async def submit(request: JobRequest, _: None = Depends(authorize)) -> dict[str, Any]:
        backend = state[f"{request.engine}_backend"]
        detail = state[f"{request.engine}_detail"]
        if backend is None or detail:
            raise HTTPException(
                status_code=503,
                detail=detail or f"{request.engine} transcription engine unavailable",
            )
        existing = jobs.get(request.id)
        if existing is not None:
            return existing
        path = _media_file(request.path)
        job = {
            "id": request.id,
            "download_id": request.download_id,
            "engine": request.engine,
            "state": "queued",
            "stage": None,
            "step": None,
            "model": request.model,
            "normalization_requested": request.normalize,
            "normalization_applied": False,
            "normalizer": NORMALIZER_NAME if request.normalize else None,
            "normalization_failure_policy": (
                request.normalization_failure_policy if request.normalize else None
            ),
            "normalization_error": None,
            "text": None,
            "duration_ms": None,
            "processed_ms": 0,
            "media_duration_ms": None,
            "progress_percent": None,
            "partial_text": "",
            "chunk_index": None,
            "chunk_count": None,
            "progress_revision": 0,
            "error": None,
        }
        jobs[request.id] = job
        try:
            queues[request.engine].put_nowait((request, path, job))
        except asyncio.QueueFull:
            jobs.pop(request.id, None)
            raise HTTPException(status_code=429, detail="transcription queue is full") from None
        log.info(
            "transcriber.job.queued job_id=%s download_id=%s engine=%s queue_depth=%d normalize=%s",
            _loggable(request.id),
            _loggable(request.download_id or "unknown"),
            request.engine,
            queues[request.engine].qsize(),
            _loggable(request.normalize),
        )
        return job

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str, _: None = Depends(authorize)) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @app.delete("/jobs/{job_id}")
    async def cancel(job_id: str, _: None = Depends(authorize)) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if job["state"] in ("queued", "running"):
            job.update(
                {
                    "state": "cancelled",
                    "stage": None,
                    "step": None,
                    "progress_revision": job["progress_revision"] + 1,
                }
            )
            execution = active_qwen.get(job_id)
            if execution is not None:
                execution.cancel()
            remember_terminal(job_id, job)
            log.info(
                "transcriber.job.cancelled job_id=%s download_id=%s",
                _loggable(job_id),
                _loggable(job.get("download_id") or "unknown"),
            )
            return job
        # DELETE doubles as an acknowledgement after the durable API has
        # stored a terminal result. This keeps the normal path near-empty,
        # while the bounded cache still supports worker restart recovery.
        response = dict(job)
        forget_job(job_id)
        return response

    return app


app = create_app()
