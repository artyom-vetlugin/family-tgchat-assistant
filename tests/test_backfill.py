"""End-to-end tests for the M3 export backfill (synthetic result.json)."""

import json
import time

import pytest

from family_assistant.backfill import run_backfill
from family_assistant.backfill.runner import norm_text
from family_assistant.store import Store
from family_assistant.transcribe import TranscriptionWorker

CHAT_ID = -1001234567890  # Bot-API form of export chat id 1234567890

VOICE_TS = 1682933696  # 2023-05-01 (UTC)


class FakeTranscriber:
    engine = "fake:test"

    def __init__(self, text="привет с дачи из голосового"):
        self.text = text
        self.calls = 0

    def transcribe(self, path):
        self.calls += 1
        return self.text


class FakeSettings:
    whisper_lang = "ru"
    job_max_attempts = 3


@pytest.fixture
def env(tmp_path):
    db_path = tmp_path / "test.db"
    store = Store(db_path)
    yield db_path, tmp_path / "media", tmp_path / "export", store
    store.close()


def write_export(export_dir, messages):
    export_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "name": "Семья",
        "type": "private_supergroup",
        "id": 1234567890,
        "messages": messages,
    }
    (export_dir / "result.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def standard_messages(export_dir):
    """The synthetic fixture: every parser/runner branch in ~7 messages."""
    voice_src = export_dir / "voice_messages/audio_1.ogg"
    voice_src.parent.mkdir(parents=True, exist_ok=True)
    voice_src.write_bytes(b"fake ogg audio")
    photo_src = export_dir / "photos/photo_1.jpg"
    photo_src.parent.mkdir(parents=True, exist_ok=True)
    photo_src.write_bytes(b"fake jpeg")
    return [
        {"id": 1, "type": "message", "date_unixtime": str(VOICE_TS - 100),
         "from": "Мама", "from_id": "user111", "text": "поехали на дачу в субботу"},
        {"id": 2, "type": "message", "date_unixtime": str(VOICE_TS - 90),
         "from": "Папа", "from_id": "user222",
         "text": ["смотри ", {"type": "link", "text": "https://example.com"}]},
        {"id": 3, "type": "message", "date_unixtime": str(VOICE_TS),
         "from": "Мама", "from_id": "user111", "text": "",
         "file": "voice_messages/audio_1.ogg", "media_type": "voice_message",
         "mime_type": "audio/ogg"},
        {"id": 4, "type": "message", "date_unixtime": str(VOICE_TS + 10),
         "from": "Папа", "from_id": "user222", "text": "",
         "photo": "photos/photo_1.jpg"},
        {"id": 5, "type": "message", "date_unixtime": str(VOICE_TS + 20),
         "from": "Папа", "from_id": "user222", "text": "",
         "file": "(File not included. Change data exporting settings to download.)",
         "media_type": "video_file"},
        {"id": 6, "type": "service", "date_unixtime": str(VOICE_TS + 30),
         "actor": "Мама", "actor_id": "user111", "action": "pin_message", "text": ""},
        {"id": 7, "type": "message", "date": "2023-05-01T12:40:00",  # no date_unixtime
         "text": "сообщение без отправителя"},
    ]


def run(env, transcribe_inline=False, worker=None):
    db_path, media_dir, export_dir, store = env
    return run_backfill(
        export_dir=export_dir, store=store, media_dir=media_dir,
        tg_chat_id=CHAT_ID, settings=FakeSettings(),
        transcribe_inline=transcribe_inline, worker=worker,
    )


def test_import_indexes_and_enqueues(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))

    report = run(env)

    assert report.total == 7
    assert report.imported == 6
    assert report.service_skipped == 1
    assert report.errors == 0
    # text searchable via FTS
    assert [r.tg_message_id for r in store.search("дачу")] == [1]
    # flattened entity text searchable
    assert [r.tg_message_id for r in store.search("example.com")] == [2]
    # voice media copied + transcribe job pending
    assert (media_dir / "export/2023/05/3.ogg").is_file()
    assert report.transcribe_jobs == 1
    assert len(store.pending_jobs("transcribe")) == 1
    # photo media row, no transcription
    photo_row = store.conn.execute(
        "SELECT m.rel_path FROM media m JOIN messages msg ON msg.id = m.message_id "
        "WHERE msg.tg_message_id = 4"
    ).fetchone()
    assert photo_row["rel_path"] == "export/2023/05/4.jpg"
    # not-included video: skipped media row
    assert report.media_skipped_notincluded == 1
    # NULL-sender message still imported
    assert store.search("без отправителя")


def test_idempotent_rerun(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))

    run(env)
    fts_count = store.conn.execute("SELECT COUNT(*) FROM search").fetchone()[0]
    report2 = run(env)

    assert report2.imported == 0
    assert report2.already_export == 6
    assert report2.errors == 0
    # the load-bearing FTS guard: no double indexing
    assert store.conn.execute("SELECT COUNT(*) FROM search").fetchone()[0] == fts_count
    # media not duplicated, file reused
    assert report2.media_copied == 0
    assert store.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0] == 3


def test_dedup_prefers_live_and_attaches_media(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    # Live rows already logged: a text message and a pre-M2 voice (no media row).
    sender_id = store.upsert_sender(111, "Мама")
    live_text = store.insert_message(
        tg_message_id=1, tg_chat_id=CHAT_ID, sender_id=sender_id,
        ts=VOICE_TS - 100, kind="text", text="поехали на дачу в субботу",
    )
    store.index_text(live_text, "поехали на дачу в субботу")
    live_voice = store.insert_message(
        tg_message_id=3, tg_chat_id=CHAT_ID, sender_id=sender_id,
        ts=VOICE_TS, kind="voice", text=None,
    )

    report = run(env)

    assert report.deduped_live == 2
    assert report.imported == 4
    # exactly one search hit — no live/export duplicate
    assert len(store.search("дачу")) == 1
    rows = store.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE tg_message_id = 1"
    ).fetchone()[0]
    assert rows == 1
    # export media attached to the live voice row that lacked it
    assert store.media_for_message(live_voice) == ("export/2023/05/3.ogg", False)
    job_ref = store.conn.execute(
        "SELECT ref_id FROM jobs WHERE job_type = 'transcribe'"
    ).fetchone()
    assert job_ref["ref_id"] == live_voice


def test_dedup_hash_fallback(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    # Same ts + same normalized text but a DIFFERENT tg_message_id (id mismatch case).
    sender_id = store.upsert_sender(111, "Мама")
    live_id = store.insert_message(
        tg_message_id=9001, tg_chat_id=CHAT_ID, sender_id=sender_id,
        ts=VOICE_TS - 100, kind="text", text="Поехали  на дачу в субботу ",
    )
    store.index_text(live_id, "Поехали  на дачу в субботу ")

    report = run(env)

    assert report.deduped_hash == 1
    assert report.deduped_live == 0
    assert len(store.search("дачу")) == 1


def test_media_reused_on_rerun(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    run(env)
    dest = media_dir / "export/2023/05/3.ogg"
    mtime = dest.stat().st_mtime_ns

    # Wipe DB but keep media files — simulates a re-import into a fresh archive.
    store.conn.execute("DELETE FROM media")
    store.conn.execute("DELETE FROM messages")
    store.conn.execute("DELETE FROM search")
    store.conn.commit()
    report2 = run(env)

    assert report2.media_reused == 2  # voice + photo found identical on disk
    assert report2.media_copied == 0
    assert dest.stat().st_mtime_ns == mtime  # file untouched


def test_media_missing_on_disk(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, [
        {"id": 1, "type": "message", "date_unixtime": "1700000000",
         "from": "Мама", "from_id": "user111", "text": "",
         "file": "voice_messages/gone.ogg", "media_type": "voice_message"},
    ])

    report = run(env)

    assert report.imported == 1
    assert report.media_missing == 1
    assert report.transcribe_jobs == 0


def test_inline_transcription(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    fake = FakeTranscriber()
    worker = TranscriptionWorker(db_path, media_dir, FakeSettings(), transcriber=fake)

    report = run(env, transcribe_inline=True, worker=worker)

    assert report.transcribed_inline == 1
    assert fake.calls == 1
    assert store.pending_jobs("transcribe") == []
    # transcript searchable, attributed to the voice message
    assert [r.tg_message_id for r in store.search("голосового")] == [3]


def test_export_sender_aliases_collected(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    store.upsert_sender(111, "Мама")  # live name known

    run(env)

    row = store.conn.execute(
        "SELECT display_name, aliases FROM senders WHERE tg_user_id = 111"
    ).fetchone()
    assert row["display_name"] == "Мама"  # not clobbered by export name
    assert row["aliases"] in (None, "[]")  # export name was identical → no alias
    # export-only sender created with the export name
    row = store.conn.execute(
        "SELECT display_name FROM senders WHERE tg_user_id = 222"
    ).fetchone()
    assert row["display_name"] == "Папа"


def test_report_render(env):
    db_path, media_dir, export_dir, store = env
    write_export(export_dir, standard_messages(export_dir))
    report = run(env)
    rendered = report.render()
    assert "imported:           6" in rendered
    assert "service skipped:    1" in rendered


# --- incremental exports: placeholder media upgraded by a fuller re-export ---

NOT_INCLUDED = "(File not included. Change data exporting settings to download.)"


def voice_msg(file_field):
    return [{
        "id": 1, "type": "message", "date_unixtime": str(VOICE_TS),
        "from": "Мама", "from_id": "user111", "text": "",
        "file": file_field, "media_type": "voice_message", "mime_type": "audio/ogg",
    }]


def test_sentinel_then_real_upgrades_media(env):
    db_path, media_dir, export_dir, store = env
    # 1st export: voice excluded from the export settings → placeholder row
    write_export(export_dir, voice_msg(NOT_INCLUDED))
    report1 = run(env)
    assert report1.media_skipped_notincluded == 1
    assert report1.transcribe_jobs == 0
    row_id = store.export_message_id(CHAT_ID, 1)
    assert store.media_for_message(row_id) == (None, True)

    # 2nd export: same message, file now included
    src = export_dir / "voice_messages/audio_1.ogg"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"real ogg audio")
    write_export(export_dir, voice_msg("voice_messages/audio_1.ogg"))
    report2 = run(env)

    assert report2.media_upgraded == 1
    assert report2.transcribe_jobs == 1
    assert store.media_for_message(row_id) == ("export/2023/05/1.ogg", False)
    assert (media_dir / "export/2023/05/1.ogg").read_bytes() == b"real ogg audio"
    assert len(store.pending_jobs("transcribe")) == 1
    # still exactly one media row — upgraded in place, not duplicated
    assert store.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0] == 1


def test_real_media_never_downgraded_by_sentinel(env):
    db_path, media_dir, export_dir, store = env
    src = export_dir / "voice_messages/audio_1.ogg"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"real ogg audio")
    write_export(export_dir, voice_msg("voice_messages/audio_1.ogg"))
    run(env)
    row_id = store.export_message_id(CHAT_ID, 1)

    # later partial export where the file is excluded again
    write_export(export_dir, voice_msg(NOT_INCLUDED))
    report2 = run(env)

    assert report2.media_upgraded == 0
    assert report2.media_skipped_notincluded == 0
    assert store.media_for_message(row_id) == ("export/2023/05/1.ogg", False)


def test_live_skipped_media_upgraded_from_export(env):
    db_path, media_dir, export_dir, store = env
    # Live row whose download was skipped (>20MB live video) — export bypasses the cap.
    sender_id = store.upsert_sender(111, "Мама")
    live_id = store.insert_message(
        tg_message_id=1, tg_chat_id=CHAT_ID, sender_id=sender_id,
        ts=VOICE_TS, kind="video", text=None,
    )
    store.insert_media(message_id=live_id, kind="video", rel_path=None,
                       bytes=50_000_000, skipped=True)
    src = export_dir / "video_files/big.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"big video bytes")
    write_export(export_dir, [{
        "id": 1, "type": "message", "date_unixtime": str(VOICE_TS),
        "from": "Мама", "from_id": "user111", "text": "",
        "file": "video_files/big.mp4", "media_type": "video_file",
        "mime_type": "video/mp4",
    }])

    report = run(env)

    assert report.deduped_live == 1
    assert report.media_upgraded == 1
    assert store.media_for_message(live_id) == ("export/2023/05/1.mp4", False)
    # video: no transcription in M3 (that's M7)
    assert report.transcribe_jobs == 0


def test_photo_upgrade_creates_no_transcribe_job(env):
    db_path, media_dir, export_dir, store = env
    photo = [{
        "id": 1, "type": "message", "date_unixtime": str(VOICE_TS),
        "from": "Папа", "from_id": "user222", "text": "",
        "photo": NOT_INCLUDED,
    }]
    write_export(export_dir, photo)
    run(env)

    src = export_dir / "photos/photo_1.jpg"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"jpeg bytes")
    photo[0]["photo"] = "photos/photo_1.jpg"
    write_export(export_dir, photo)
    report2 = run(env)

    assert report2.media_upgraded == 1
    assert report2.transcribe_jobs == 0
    row_id = store.export_message_id(CHAT_ID, 1)
    assert store.media_for_message(row_id) == ("export/2023/05/1.jpg", False)


def test_norm_text():
    assert norm_text("  Привет,   МИР \n") == "привет, мир"


def test_derive_botapi_chat_id():
    from family_assistant.backfill.__main__ import derive_botapi_chat_id

    assert derive_botapi_chat_id(1234567890, "private_supergroup") == -1001234567890
    assert derive_botapi_chat_id(987654, "private_group") == -987654
