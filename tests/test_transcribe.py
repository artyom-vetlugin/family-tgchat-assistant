"""Worker tests with a fake transcriber — no whisper model needed."""

import time

import pytest

from family_assistant.store import Store
from family_assistant.transcribe import TranscriptionWorker


class FakeTranscriber:
    engine = "fake:test"

    def __init__(self, text="завтра едем на дачу", fail_times=0):
        self.text = text
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("decode failed")
        return self.text


class FakeSettings:
    whisper_lang = "ru"
    job_max_attempts = 3


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "test.db"
    media_dir = tmp_path / "media"
    store = Store(db_path)
    yield db_path, media_dir, store
    store.close()


def make_voice_message(store, media_dir, msg_id=1):
    sender_id = store.upsert_sender(1, "Папа")
    row_id = store.insert_message(
        tg_message_id=msg_id, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="voice", text=None,
    )
    rel_path = f"live/2026/06/{msg_id}.oga"
    abs_path = media_dir / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"fake audio")
    store.insert_media(message_id=row_id, kind="voice", rel_path=rel_path, bytes=10)
    job_id = store.create_job(job_type="transcribe", ref_id=row_id)
    return row_id, job_id


def run_one(worker, store, job_id):
    """Drive the worker's processing synchronously (no thread)."""
    worker._process(store, job_id)


def test_worker_processes_job(env):
    db_path, media_dir, store = env
    row_id, job_id = make_voice_message(store, media_dir)
    fake = FakeTranscriber()
    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=fake)

    run_one(worker, store, job_id)

    # transcript stored, indexed, job done
    rows = store.search("дачу")
    assert [r.tg_message_id for r in rows] == [1]
    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "done"


def test_worker_retry_then_error(env):
    db_path, media_dir, store = env
    row_id, job_id = make_voice_message(store, media_dir)
    fake = FakeTranscriber(fail_times=99)  # always fails
    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=fake)

    for _ in range(3):
        run_one(worker, store, job_id)

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "error"
    assert fake.calls == 3
    # retries were re-enqueued by _process until attempts ran out
    assert worker.queue.qsize() == 2


def test_worker_recovers_after_transient_failure(env):
    db_path, media_dir, store = env
    row_id, job_id = make_voice_message(store, media_dir)
    fake = FakeTranscriber(fail_times=1)  # fail once, then succeed
    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=fake)

    run_one(worker, store, job_id)  # fails -> pending
    run_one(worker, store, job_id)  # succeeds

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "done"
    assert store.search("дачу")


def test_worker_startup_recovery(env):
    db_path, media_dir, store = env
    # one job left pending, one stuck inflight (simulated crash)
    _, job1 = make_voice_message(store, media_dir, msg_id=1)
    _, job2 = make_voice_message(store, media_dir, msg_id=2)
    store.claim_job(job2)  # inflight

    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=FakeTranscriber())
    worker.recover(store)

    queued = {worker.queue.get_nowait(), worker.queue.get_nowait()}
    assert queued == {job1, job2}


def test_missing_media_file_errors_out(env):
    db_path, media_dir, store = env
    sender_id = store.upsert_sender(1, "Папа")
    row_id = store.insert_message(
        tg_message_id=9, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="voice", text=None,
    )
    job_id = store.create_job(job_type="transcribe", ref_id=row_id)  # no media row
    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=FakeTranscriber())

    for _ in range(3):
        run_one(worker, store, job_id)

    state = store.conn.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()[0]
    assert state == "error"
