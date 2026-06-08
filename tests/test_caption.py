"""Caption worker tests with a fake captioner — no API calls, no network."""

import base64
import io
import time

import pytest
from PIL import Image

from family_assistant.caption import CaptionWorker, encode_image
from family_assistant.store import Store


class FakeCaptioner:
    model = "fake:test"

    def __init__(self, text="семья на даче у крыльца", fail_times=0):
        self.text = text
        self.fail_times = fail_times
        self.calls = 0

    def caption(self, path):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("api error")
        return self.text


class FakeSettings:
    job_max_attempts = 3


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "test.db"
    media_dir = tmp_path / "media"
    store = Store(db_path)
    yield db_path, media_dir, store
    store.close()


def write_jpeg(path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(120, 30, 30)).save(path, format="JPEG")


def make_photo_media(store, media_dir, msg_id=1):
    """Photo message + downloaded media row + caption job; returns (media_id, job_id)."""
    sender_id = store.upsert_sender(1, "Мама")
    row_id = store.insert_message(
        tg_message_id=msg_id, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="photo", text=None,
    )
    rel_path = f"live/2026/06/{msg_id}.jpg"
    write_jpeg(media_dir / rel_path)
    media_id = store.insert_media(message_id=row_id, kind="photo", rel_path=rel_path)
    job_id = store.create_job(job_type="caption", ref_id=media_id)
    return media_id, job_id


def run_one(worker, store, job_id):
    """Drive the worker's processing synchronously (no thread)."""
    worker._process(store, job_id)


def test_worker_captions_photo_and_indexes_it(env):
    db_path, media_dir, store = env
    media_id, job_id = make_photo_media(store, media_dir)
    worker = CaptionWorker(db_path, media_dir, FakeSettings(), captioner=FakeCaptioner())

    run_one(worker, store, job_id)

    # caption stored, searchable via FTS, job done — the M4 acceptance path
    rows = store.search("даче")
    assert [r.tg_message_id for r in rows] == [1]
    assert rows[0].kind == "photo"
    # the caption is surfaced as the row body for the agent
    assert rows[0].text == "семья на даче у крыльца"
    assert "[фото]" in rows[0].format()
    caption = store.conn.execute(
        "SELECT text, model FROM captions WHERE media_id = ?", (media_id,)
    ).fetchone()
    assert caption["text"] == "семья на даче у крыльца"
    assert caption["model"] == "fake:test"
    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "done"


def test_worker_retry_then_error(env):
    db_path, media_dir, store = env
    _, job_id = make_photo_media(store, media_dir)
    fake = FakeCaptioner(fail_times=99)  # always fails
    worker = CaptionWorker(db_path, media_dir, FakeSettings(), captioner=fake)

    for _ in range(3):
        run_one(worker, store, job_id)

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "error"
    assert fake.calls == 3
    assert worker.queue.qsize() == 2  # re-enqueued until attempts ran out


def test_worker_recovers_after_transient_failure(env):
    db_path, media_dir, store = env
    _, job_id = make_photo_media(store, media_dir)
    worker = CaptionWorker(
        db_path, media_dir, FakeSettings(), captioner=FakeCaptioner(fail_times=1)
    )

    run_one(worker, store, job_id)  # fails -> pending
    run_one(worker, store, job_id)  # succeeds

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "done"
    assert store.search("даче")


def test_worker_startup_recovery_leaves_batched_jobs_alone(env):
    db_path, media_dir, store = env
    _, job1 = make_photo_media(store, media_dir, msg_id=1)
    _, job2 = make_photo_media(store, media_dir, msg_id=2)
    _, job3 = make_photo_media(store, media_dir, msg_id=3)
    store.claim_job(job2)  # stale inflight (simulated crash)
    store.claim_job(job3)  # inflight at the Batch API — must NOT be stolen
    store.set_job_batch(job3, "batch_abc")

    worker = CaptionWorker(db_path, media_dir, FakeSettings(), captioner=FakeCaptioner())
    worker.recover(store)

    queued = {worker.queue.get_nowait(), worker.queue.get_nowait()}
    assert queued == {job1, job2}
    assert worker.queue.empty()
    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job3,)).fetchone()[0]
    assert state == "inflight"


def test_missing_media_file_errors_out(env):
    db_path, media_dir, store = env
    sender_id = store.upsert_sender(1, "Мама")
    row_id = store.insert_message(
        tg_message_id=9, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="photo", text=None,
    )
    media_id = store.insert_media(message_id=row_id, kind="photo", rel_path=None, skipped=True)
    job_id = store.create_job(job_type="caption", ref_id=media_id)
    worker = CaptionWorker(db_path, media_dir, FakeSettings(), captioner=FakeCaptioner())

    for _ in range(3):
        run_one(worker, store, job_id)

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "error"


def test_encode_image_resizes_and_reencodes(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (2048, 512), color=(0, 100, 0)).save(src, format="PNG")

    b64 = encode_image(src, long_edge=1024, quality=80)

    out = Image.open(io.BytesIO(base64.standard_b64decode(b64)))
    assert out.format == "JPEG"
    assert max(out.size) <= 1024
    assert out.size == (1024, 256)  # aspect ratio preserved


def test_encode_image_corrupt_file_raises(tmp_path):
    src = tmp_path / "broken.jpg"
    src.write_bytes(b"definitely not an image")
    with pytest.raises(Exception):
        encode_image(src)
