"""Video worker tests (M7) — fake transcriber + captioner, no whisper/PyAV/API."""

import time
from types import SimpleNamespace

import pytest

from family_assistant import video as video_mod
from family_assistant.store import Store
from family_assistant.video import VideoWorker, build_video_text


class FakeTranscriber:
    engine = "fake:test"

    def __init__(self, text="на видео мы поём песню", fail_times=0):
        self.text = text
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("decode failed")
        return self.text


class FakeCaptioner:
    model = "fake:caption"

    def __init__(self, text="торт со свечами"):
        self.text = text
        self.last_usage = None

    def caption_pil(self, img):
        self.last_usage = SimpleNamespace(
            input_tokens=50, cache_read_input_tokens=0,
            cache_creation_input_tokens=0, output_tokens=5,
        )
        return self.text


class FakeSettings:
    whisper_lang = "ru"
    job_max_attempts = 3
    video_keyframes = 1


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "test.db"
    media_dir = tmp_path / "media"
    store = Store(db_path)
    yield db_path, media_dir, store
    store.close()


def make_video_message(store, media_dir, msg_id=1, skipped=False):
    sender_id = store.upsert_sender(1, "Папа")
    row_id = store.insert_message(
        tg_message_id=msg_id, tg_chat_id=-100, sender_id=sender_id,
        ts=int(time.time()), kind="video", text=None,
    )
    rel_path = None if skipped else f"live/2026/06/{msg_id}.mp4"
    if rel_path:
        abs_path = media_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(b"fake video")
    store.insert_media(
        message_id=row_id, kind="video", rel_path=rel_path, bytes=100, skipped=skipped
    )
    job_id = store.create_job(job_type="video", ref_id=row_id)
    return row_id, job_id


def make_worker(db_path, media_dir, transcriber, captioner, embeds=None):
    return VideoWorker(
        db_path, media_dir, FakeSettings(),
        transcriber=transcriber, captioner=captioner,
        embed_enqueue=(embeds.append if embeds is not None else None),
    )


# --- build_video_text ------------------------------------------------------


def test_build_video_text_combines():
    assert build_video_text("привет", ["торт", "свечи"]) == "привет | в кадре: торт; свечи"
    assert build_video_text("привет", []) == "привет"
    assert build_video_text("", ["торт"]) == "в кадре: торт"
    assert build_video_text("", []) == ""


# --- worker ----------------------------------------------------------------


def test_worker_indexes_transcript_and_caption(env, monkeypatch):
    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: ["frame"])
    row_id, job_id = make_video_message(store, media_dir)
    embeds: list[int] = []
    worker = make_worker(db_path, media_dir, FakeTranscriber(), FakeCaptioner(), embeds)

    worker._process(store, job_id)

    # spoken word AND caption word both searchable, with [видео] prefix
    assert [r.tg_message_id for r in store.search("поём")] == [1]
    caption_hits = store.search("торт")
    assert [r.tg_message_id for r in caption_hits] == [1]
    assert caption_hits[0].format().startswith("[")
    assert "[видео]" in caption_hits[0].format()
    # embed job enqueued, job done
    assert len(embeds) == 1
    assert store.job_state(job_id)[0] == "done"


def test_worker_records_caption_spend(env, monkeypatch):
    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: ["frame"])
    _, job_id = make_video_message(store, media_dir)
    worker = make_worker(db_path, media_dir, FakeTranscriber(), FakeCaptioner())

    worker._process(store, job_id)

    rows = store.spend_summary(time.strftime("%Y-%m"))
    assert any(r["model"] == "fake:caption" and r["out_tokens"] == 5 for r in rows)


def test_worker_transcript_only_when_no_frames(env, monkeypatch):
    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: [])
    _, job_id = make_video_message(store, media_dir)
    worker = make_worker(db_path, media_dir, FakeTranscriber(), FakeCaptioner())

    worker._process(store, job_id)

    assert [r.tg_message_id for r in store.search("поём")] == [1]
    assert store.job_state(job_id)[0] == "done"


def test_worker_retries_then_errors(env, monkeypatch):
    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: ["frame"])
    _, job_id = make_video_message(store, media_dir)
    worker = make_worker(db_path, media_dir, FakeTranscriber(fail_times=99), FakeCaptioner())

    for _ in range(FakeSettings.job_max_attempts):
        worker._process(store, job_id)
    assert store.job_state(job_id)[0] == "error"


def test_worker_skipped_video_errors(env, monkeypatch):
    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: ["frame"])
    _, job_id = make_video_message(store, media_dir, skipped=True)
    worker = make_worker(db_path, media_dir, FakeTranscriber(), FakeCaptioner())

    worker._process(store, job_id)
    # >20MB skipped video has no file → finished as failed (pending until retries)
    assert store.job_state(job_id)[0] in ("pending", "error")
    assert not store.search("поём")


# --- backfill --------------------------------------------------------------


def test_run_video_backfill_drains_export_videos(env, monkeypatch):
    from family_assistant.video_backfill import run_video_backfill

    db_path, media_dir, store = env
    monkeypatch.setattr(video_mod, "extract_keyframes", lambda path, n: ["frame"])
    # two export videos with files but no job/transcript yet
    make_video_message(store, media_dir, msg_id=10)
    make_video_message(store, media_dir, msg_id=11)
    # clear the jobs make_video_message created so backfill enqueues them itself
    store.conn.execute("DELETE FROM jobs")
    store.conn.commit()

    worker = make_worker(db_path, media_dir, FakeTranscriber(), FakeCaptioner())
    report = run_video_backfill(store=store, settings=FakeSettings(), worker=worker)

    assert report.videos_without_transcript == 2
    assert report.done == 2
    assert report.pending_left == 0
    # re-run is a no-op (both now have transcripts)
    report2 = run_video_backfill(store=store, settings=FakeSettings(), worker=worker)
    assert report2.enqueued == 0
