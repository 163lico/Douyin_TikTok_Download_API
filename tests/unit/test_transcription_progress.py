"""Ephemeral transcription progress stays bounded and parseable."""

from __future__ import annotations

from typing import Any

import pytest

from dtk.services import transcription_progress


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, int] = {}

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = value
        self.expiries[key] = ex

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)


async def test_progress_round_trips_through_redis_with_a_short_ttl(monkeypatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(transcription_progress, "get_redis", lambda: redis)
    transcription_id = "7afe9635-9deb-4590-b76a-46a18c38bf7b"
    job: dict[str, Any] = {
        "state": "running",
        "stage": "recognizing",
        "step": "external_request",
        "progress_revision": 4,
        "processed_ms": 500,
        "media_duration_ms": 1000,
        "progress_percent": 50.0,
        "chunk_index": 1,
        "chunk_count": 2,
        "partial_text": "temporary text",
        # The final text and errors are deliberately outside the snapshot.
        "text": "must not be copied",
        "error": "must not be copied",
    }

    await transcription_progress.publish(transcription_id, job)
    snapshot = await transcription_progress.read(transcription_id)

    assert snapshot is not None
    assert snapshot.stage == "recognizing"
    assert snapshot.step == "external_request"
    assert snapshot.revision == 4
    assert snapshot.partial_text == "temporary text"
    assert snapshot.chunk_index == 1
    assert snapshot.chunk_count == 2
    key = transcription_progress.KEY.format(transcription_id=transcription_id)
    assert redis.expiries[key] == transcription_progress.TTL_SECONDS
    assert "must not be copied" not in redis.values[key]

    await transcription_progress.clear(transcription_id)
    assert await transcription_progress.read(transcription_id) is None


async def test_malformed_progress_is_ignored(monkeypatch) -> None:
    redis = FakeRedis()
    transcription_id = "7afe9635-9deb-4590-b76a-46a18c38bf7b"
    redis.values[transcription_progress.KEY.format(transcription_id=transcription_id)] = "not-json"
    monkeypatch.setattr(transcription_progress, "get_redis", lambda: redis)

    assert await transcription_progress.read(transcription_id) is None


async def test_queued_progress_has_no_stage_or_step(monkeypatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(transcription_progress, "get_redis", lambda: redis)
    transcription_id = "7afe9635-9deb-4590-b76a-46a18c38bf7b"

    await transcription_progress.publish(
        transcription_id,
        {
            "state": "queued",
            "stage": None,
            "step": None,
            "progress_revision": 0,
        },
    )

    snapshot = await transcription_progress.read(transcription_id)
    assert snapshot is not None
    assert snapshot.stage is None
    assert snapshot.step is None


@pytest.mark.parametrize(
    ("state", "stage", "step"),
    [
        ("running", None, None),
        ("running", "recognizing", "normalizing_text"),
        ("done", "post_processing", "normalizing_text"),
    ],
)
def test_invalid_state_stage_step_combinations_are_rejected(state, stage, step) -> None:
    with pytest.raises(ValueError):
        transcription_progress.from_job({"state": state, "stage": stage, "step": step})
