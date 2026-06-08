"""Batch-API caption backfill tests with a fake Anthropic client."""

import time
from types import SimpleNamespace

import pytest
from PIL import Image

from family_assistant.caption_backfill import run_caption_backfill
from family_assistant.store import Store


class FakeSettings:
    anthropic_api_key = "test"
    caption_model = "claude-haiku-4-5"
    caption_max_tokens = 256
    caption_batch_chunk = 500
    caption_poll_seconds = 0
    caption_resize_long_edge = 1024
    caption_jpeg_quality = 80
    job_max_attempts = 3


def _success_result(custom_id, text):
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=[SimpleNamespace(type="text", text=text)]),
        ),
    )


def _error_result(custom_id):
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


class FakeBatches:
    """Stub of client.messages.batches: each create() succeeds instantly and
    yields one result per request — succeeded by default, errored for the
    custom_ids listed in fail_ids."""

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.created: list[list[dict]] = []
        self._results: dict[str, list] = {}

    def create(self, *, requests):
        batch_id = f"batch_{len(self.created)}"
        self.created.append(requests)
        self._results[batch_id] = [
            _error_result(r["custom_id"])
            if r["custom_id"] in self.fail_ids
            else _success_result(r["custom_id"], "дети на даче в саду")
            for r in requests
        ]
        return SimpleNamespace(id=batch_id, processing_status="in_progress")

    def retrieve(self, batch_id):
        return SimpleNamespace(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        yield from self._results[batch_id]

    def preload(self, batch_id, results):
        """Register results for a batch this client never created (resume tests)."""
        self._results[batch_id] = results


def make_client(batches):
    return SimpleNamespace(messages=SimpleNamespace(batches=batches))


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "test.db"
    media_dir = tmp_path / "media"
    store = Store(db_path)
    yield media_dir, store
    store.close()


def add_photo(store, media_dir, msg_id, *, write_file=True):
    sender_id = store.upsert_sender(1, "Папа")
    row_id = store.insert_message(
        tg_message_id=msg_id, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="photo", text=None,
    )
    rel_path = f"export/2024/05/{msg_id}.jpg"
    abs_path = media_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    if write_file:
        Image.new("RGB", (64, 48), color=(50, 80, 50)).save(abs_path, format="JPEG")
    else:
        abs_path.write_bytes(b"not an image")
    return store.insert_media(message_id=row_id, kind="photo", rel_path=rel_path)


def job_row(store, media_id):
    return store.conn.execute(
        "SELECT * FROM jobs WHERE job_type = 'caption' AND ref_id = ?", (media_id,)
    ).fetchone()


def test_backfill_captions_and_indexes_photos(env):
    media_dir, store = env
    m1 = add_photo(store, media_dir, msg_id=1)
    m2 = add_photo(store, media_dir, msg_id=2)
    batches = FakeBatches()

    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=FakeSettings(),
        client=make_client(batches),
    )

    assert report.photos_uncaptioned == 2
    assert report.batches == 1
    assert report.submitted == 2
    assert report.succeeded == 2
    assert report.indexed == 2
    rows = store.search("даче")
    assert sorted(r.tg_message_id for r in rows) == [1, 2]
    for media_id in (m1, m2):
        row = job_row(store, media_id)
        assert row["state"] == "done"
        assert row["batch_id"] == "batch_0"
    # request shape: vision block + frozen system prompt
    req = batches.created[0][0]
    assert req["params"]["model"] == "claude-haiku-4-5"
    assert req["params"]["messages"][0]["content"][0]["type"] == "image"


def test_rerun_is_noop(env):
    media_dir, store = env
    add_photo(store, media_dir, msg_id=1)
    settings = FakeSettings()
    run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings,
        client=make_client(FakeBatches()),
    )

    second = FakeBatches()
    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings, client=make_client(second),
    )

    assert report.photos_uncaptioned == 0
    assert report.submitted == 0
    assert second.created == []
    # no double-indexing in FTS
    fts = store.conn.execute("SELECT COUNT(*) FROM search").fetchone()[0]
    assert fts == 1


def test_resume_polls_inflight_batch_instead_of_resubmitting(env):
    media_dir, store = env
    media_id = add_photo(store, media_dir, msg_id=1)
    # Simulate a killed previous run: job claimed + batch submitted upstream.
    job_id = store.create_job(job_type="caption", ref_id=media_id)
    store.claim_job(job_id)
    store.set_job_batch(job_id, "batch_old")
    batches = FakeBatches()
    batches.preload(
        "batch_old", [_success_result(f"caption-{job_id}", "торт со свечами")]
    )

    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=FakeSettings(),
        client=make_client(batches),
    )

    assert batches.created == []  # nothing re-submitted
    assert report.succeeded == 1
    assert store.search("свечами")
    assert job_row(store, media_id)["state"] == "done"


def test_errored_results_are_retried_until_attempts_exhausted(env):
    media_dir, store = env
    media_id = add_photo(store, media_dir, msg_id=1)
    job_id = store.create_job(job_type="caption", ref_id=media_id)
    batches = FakeBatches(fail_ids={f"caption-{job_id}"})

    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=FakeSettings(),
        client=make_client(batches),
    )

    # resubmitted in a fresh batch each round until max_attempts
    assert len(batches.created) == 3
    assert report.errored == 3
    assert job_row(store, media_id)["state"] == "error"
    assert store.conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0] == 0


def test_retry_errored_recovers_failed_jobs(env):
    media_dir, store = env
    media_id = add_photo(store, media_dir, msg_id=1)
    job_id = store.create_job(job_type="caption", ref_id=media_id)
    settings = FakeSettings()

    # Run 1: every batch result errors -> job exhausts attempts and goes 'error'.
    run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings,
        client=make_client(FakeBatches(fail_ids={f"caption-{job_id}"})),
    )
    assert job_row(store, media_id)["state"] == "error"

    # Run 2 WITHOUT retry: the dead job stays dead, nothing is re-submitted.
    second = FakeBatches()
    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings, client=make_client(second),
    )
    assert second.created == []
    assert report.errored_retried == 0
    assert job_row(store, media_id)["state"] == "error"

    # Run 3 WITH retry: the job is revived, re-submitted, and succeeds.
    third = FakeBatches()
    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings,
        client=make_client(third), retry_errored=True,
    )
    assert report.errored_retried == 1
    assert report.succeeded == 1
    assert job_row(store, media_id)["state"] == "done"
    assert store.search("даче")
    assert store.conn.execute("SELECT COUNT(*) FROM captions").fetchone()[0] == 1


def test_corrupt_image_is_skipped_without_api_call(env):
    media_dir, store = env
    add_photo(store, media_dir, msg_id=1, write_file=False)  # garbage bytes
    good = add_photo(store, media_dir, msg_id=2)
    batches = FakeBatches()

    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=FakeSettings(),
        client=make_client(batches),
    )

    assert report.skipped_encode == 1
    assert report.succeeded == 1
    assert len(batches.created) == 1
    assert len(batches.created[0]) == 1  # only the good photo went to the API
    assert job_row(store, good)["state"] == "done"


def test_max_batches_caps_the_run(env):
    media_dir, store = env
    for i in range(3):
        add_photo(store, media_dir, msg_id=i + 1)
    settings = FakeSettings()
    settings.caption_batch_chunk = 2  # 3 photos -> 2 batches needed
    batches = FakeBatches()

    report = run_caption_backfill(
        store=store, media_dir=media_dir, settings=settings,
        client=make_client(batches), max_batches=1,
    )

    assert report.batches == 1
    assert report.succeeded == 2
    assert report.pending_left == 1
