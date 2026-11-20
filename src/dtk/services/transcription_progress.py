"""Ephemeral progress snapshots for an in-flight transcription.

The durable transcription row contains only the authoritative final text.
Provisional ASR fragments cross the worker/API process boundary through Redis
and expire quickly if either process disappears.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Final

from dtk.core.redis import get_redis

KEY: Final = "transcriptions:progress:{transcription_id}"
TTL_SECONDS: Final = 120
RUNNING_STATE: Final = "running"
TERMINAL_OR_QUEUED_STATES: Final = frozenset({"queued", "done", "failed", "cancelled"})
STEPS_BY_STAGE: Final = {
    "preparing_audio": frozenset({"decoding_audio", "extracting_and_splitting_audio"}),
    "recognizing": frozenset({"detecting_speech", "local_inference", "external_request"}),
    "post_processing": frozenset({"merging_results", "adding_punctuation", "normalizing_text"}),
}


@dataclass(frozen=True, slots=True)
class TranscriptionProgress:
    state: str
    stage: str | None
    step: str | None
    revision: int
    processed_ms: int
    duration_ms: int | None
    progress_percent: float | None
    chunk_index: int | None
    chunk_count: int | None
    partial_text: str


def _identifier(value: uuid.UUID | str) -> str:
    return str(value)


def _phase(
    state_value: object,
    stage_value: object,
    step_value: object,
) -> tuple[str, str | None, str | None]:
    state = str(state_value)
    stage = str(stage_value) if stage_value is not None else None
    step = str(step_value) if step_value is not None else None
    if state == RUNNING_STATE:
        if stage not in STEPS_BY_STAGE or step not in STEPS_BY_STAGE[stage]:
            raise ValueError("running transcription has an invalid stage/step pair")
    elif state in TERMINAL_OR_QUEUED_STATES:
        if stage is not None or step is not None:
            raise ValueError("non-running transcription must not carry stage or step")
    else:
        raise ValueError("transcription has an invalid state")
    return state, stage, step


def from_job(job: dict[str, Any]) -> TranscriptionProgress:
    """Reduce the sidecar's internal job response to the safe progress contract."""
    duration = int(job.get("media_duration_ms") or 0) or None
    percent = job.get("progress_percent")
    state, stage, step = _phase(job.get("state"), job.get("stage"), job.get("step"))
    return TranscriptionProgress(
        state=state,
        stage=stage,
        step=step,
        revision=max(0, int(job.get("progress_revision") or 0)),
        processed_ms=max(0, int(job.get("processed_ms") or 0)),
        duration_ms=duration,
        progress_percent=float(percent) if percent is not None else None,
        chunk_index=(int(job["chunk_index"]) if job.get("chunk_index") is not None else None),
        chunk_count=(int(job["chunk_count"]) if job.get("chunk_count") is not None else None),
        partial_text=str(job.get("partial_text") or ""),
    )


async def publish(
    transcription_id: uuid.UUID | str,
    job: dict[str, Any],
    *,
    ttl: int = TTL_SECONDS,
) -> None:
    snapshot = from_job(job)
    await get_redis().set(
        KEY.format(transcription_id=_identifier(transcription_id)),
        json.dumps(asdict(snapshot), ensure_ascii=False, separators=(",", ":")),
        ex=max(1, ttl),
    )


async def read(transcription_id: uuid.UUID | str) -> TranscriptionProgress | None:
    raw = await get_redis().get(KEY.format(transcription_id=_identifier(transcription_id)))
    if not raw:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        state, stage, step = _phase(value.get("state"), value.get("stage"), value.get("step"))
        return TranscriptionProgress(
            state=state,
            stage=stage,
            step=step,
            revision=max(0, int(value.get("revision") or 0)),
            processed_ms=max(0, int(value.get("processed_ms") or 0)),
            duration_ms=(int(value["duration_ms"]) if value.get("duration_ms") else None),
            progress_percent=(
                float(value["progress_percent"])
                if value.get("progress_percent") is not None
                else None
            ),
            chunk_index=(
                int(value["chunk_index"]) if value.get("chunk_index") is not None else None
            ),
            chunk_count=(
                int(value["chunk_count"]) if value.get("chunk_count") is not None else None
            ),
            partial_text=str(value.get("partial_text") or ""),
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


async def clear(transcription_id: uuid.UUID | str) -> None:
    await get_redis().delete(KEY.format(transcription_id=_identifier(transcription_id)))


__all__ = [
    "KEY",
    "TTL_SECONDS",
    "TranscriptionProgress",
    "clear",
    "from_job",
    "publish",
    "read",
]
