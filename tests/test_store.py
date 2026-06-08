"""Smoke tests for the SQLite store: schema, inserts, Russian trigram search."""

import time

import pytest

from family_assistant.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def add_message(store, *, msg_id, text, sender="Мама", ts=None, kind="text"):
    sender_id = store.upsert_sender(hash(sender) % 10**9, sender)
    row_id = store.insert_message(
        tg_message_id=msg_id,
        tg_chat_id=-100,
        sender_id=sender_id,
        ts=ts or int(time.time()),
        kind=kind,
        text=text,
    )
    if row_id and text:
        store.index_text(row_id, text)
    return row_id


def test_insert_and_dedup(store):
    assert add_message(store, msg_id=1, text="привет") is not None
    assert add_message(store, msg_id=1, text="привет") is None  # duplicate


def test_russian_trigram_search(store):
    add_message(store, msg_id=1, text="Мы поехали на дачу в субботу")
    add_message(store, msg_id=2, text="Купили молоко и хлеб")
    add_message(store, msg_id=3, text="Поездка на море этим летом")

    # Stem 'поезд' must match both 'поехали'... no wait — trigram is substring:
    # 'поезд' matches 'поездка' but NOT 'поехали'. Verify substring semantics.
    rows = store.search("поездк")
    assert [r.tg_message_id for r in rows] == [3]

    rows = store.search("поехали")
    assert [r.tg_message_id for r in rows] == [1]

    # Case-insensitive
    rows = store.search("МОЛОКО")
    assert [r.tg_message_id for r in rows] == [2]


def test_messages_between_window_and_resolution(store):
    day = int(time.mktime(time.strptime("2024-05-10", "%Y-%m-%d")))
    add_message(store, msg_id=1, text="до окна", ts=day - 1)
    add_message(store, msg_id=2, text="внутри окна", ts=day + 100)
    # a photo whose searchable text lives in captions, not messages.text
    sender_id = store.upsert_sender(7, "Папа")
    pid = store.insert_message(
        tg_message_id=3, tg_chat_id=-100, sender_id=sender_id,
        ts=day + 200, kind="photo", text=None,
    )
    media_id = store.insert_media(message_id=pid, kind="photo", rel_path="x.jpg")
    store.insert_caption(media_id=media_id, text="дети на даче", model="m")
    add_message(store, msg_id=4, text="после окна", ts=day + 86400)

    rows = store.messages_between(-100, day, day + 86400)  # [day, next day)
    assert [r.tg_message_id for r in rows] == [2, 3]
    photo = next(r for r in rows if r.tg_message_id == 3)
    assert photo.text == "дети на даче"  # caption resolved via COALESCE


def test_first_message_date(store):
    assert store.first_message_date(-100) is None
    add_message(store, msg_id=1, text="второе", ts=200)
    add_message(store, msg_id=2, text="первое", ts=100)
    assert store.first_message_date(-100) == 100


def test_search_too_short_query_is_safe(store):
    add_message(store, msg_id=1, text="привет")
    assert store.search("пр") == []  # <3 chars for trigram — no crash


def test_search_quotes_are_safe(store):
    add_message(store, msg_id=1, text='он сказал "привет" и ушёл')
    rows = store.search('"привет"')
    assert [r.tg_message_id for r in rows] == [1]


def test_recent_window_and_around(store):
    now = int(time.time())
    for i in range(10):
        add_message(store, msg_id=i + 1, text=f"сообщение номер {i + 1}", ts=now - (10 - i) * 60)

    recent = store.recent_window(-100, hours=1)
    assert len(recent) == 10
    assert recent[0].tg_message_id == 1  # chronological order

    around = store.get_messages_around(-100, 5, n=2)
    ids = [r.tg_message_id for r in around]
    assert 5 in ids and 4 in ids and 6 in ids
    assert len(ids) == 5  # 2 before + anchor + 2 after


def test_search_filters(store):
    now = int(time.time())
    add_message(store, msg_id=1, text="дача весной", sender="Мама", ts=now - 86400 * 30)
    add_message(store, msg_id=2, text="дача летом", sender="Папа", ts=now)

    rows = store.search("дача", sender="Папа")
    assert [r.tg_message_id for r in rows] == [2]

    rows = store.search("дача", after_ts=now - 86400)
    assert [r.tg_message_id for r in rows] == [2]


def test_transcript_indexing_searchable(store):
    """Media message with no text; transcript indexed later must be findable."""
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    store.index_text(row_id, "завтра идём к врачу в десять утра")
    rows = store.search("врачу")
    assert [r.tg_message_id for r in rows] == [1]


# --- M2: media / transcripts / jobs ----------------------------------------


def test_job_lifecycle(store):
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    job_id = store.create_job(job_type="transcribe", ref_id=row_id)

    # duplicate create returns the same job
    assert store.create_job(job_type="transcribe", ref_id=row_id) == job_id

    assert store.pending_jobs("transcribe") == [job_id]
    assert store.job_ref(job_id) == ("transcribe", row_id)

    attempts = store.claim_job(job_id)
    assert attempts == 1
    assert store.pending_jobs("transcribe") == []

    assert store.finish_job(job_id, ok=True, max_attempts=3) == "done"


def test_job_retry_then_error(store):
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    job_id = store.create_job(job_type="transcribe", ref_id=row_id)

    store.claim_job(job_id)  # attempts=1
    assert store.finish_job(job_id, ok=False, max_attempts=3) == "pending"
    store.claim_job(job_id)  # attempts=2
    assert store.finish_job(job_id, ok=False, max_attempts=3) == "pending"
    store.claim_job(job_id)  # attempts=3 == max
    assert store.finish_job(job_id, ok=False, max_attempts=3) == "error"
    assert store.pending_jobs("transcribe") == []


def test_reset_stale_jobs(store):
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    job_id = store.create_job(job_type="transcribe", ref_id=row_id)
    store.claim_job(job_id)  # inflight, simulating a crash mid-job

    assert store.reset_stale_jobs("transcribe") == 1
    assert store.pending_jobs("transcribe") == [job_id]


def test_insert_media_and_lookup(store):
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    store.insert_media(message_id=row_id, kind="voice", rel_path="live/2026/06/1.oga", bytes=1234)
    assert store.media_for_message(row_id) == ("live/2026/06/1.oga", False)


def test_insert_media_skipped(store):
    row_id = add_message(store, msg_id=1, text=None, kind="video")
    store.insert_media(message_id=row_id, kind="video", rel_path=None, bytes=50_000_000, skipped=True)
    assert store.media_for_message(row_id) == (None, True)


def test_transcript_visible_in_all_retrieval_paths(store):
    """Regression: voice transcripts must appear in tool output, not '[voice без текста]'.

    The bug: retrieval queries selected messages.text (NULL for voice) without
    joining transcripts, so the agent saw transcribed messages as empty."""
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    store.insert_transcript(message_id=row_id, text="встречаемся у бабушки в шесть", engine="fake")
    store.index_text(row_id, "встречаемся у бабушки в шесть")
    add_message(store, msg_id=2, text="ок, понял")

    # fts_search path
    rows = store.search("бабушки")
    assert rows and "бабушки" in rows[0].text and "[голосовое]" in rows[0].format()

    # recent_window path
    rows = store.recent_window(-100, hours=1)
    voice_row = next(r for r in rows if r.tg_message_id == 1)
    assert voice_row.text and "бабушки" in voice_row.text

    # get_messages_around path
    rows = store.get_messages_around(-100, 2, n=2)
    voice_row = next(r for r in rows if r.tg_message_id == 1)
    assert voice_row.text and "бабушки" in voice_row.text


# --- M3: backfill helpers ----------------------------------------------------


def test_upsert_export_sender_accumulates_aliases(store):
    live_id = store.upsert_sender(123, "Мама")
    # Export name variant becomes an alias; display_name stays the live one.
    assert store.upsert_export_sender(123, "Мария Иванова") == live_id
    assert store.upsert_export_sender(123, "Мария Иванова") == live_id  # no dup
    assert store.upsert_export_sender(123, "Мама") == live_id  # == display_name, skip
    row = store.conn.execute("SELECT display_name, aliases FROM senders WHERE id = ?", (live_id,)).fetchone()
    assert row["display_name"] == "Мама"
    assert row["aliases"] == '["Мария Иванова"]'


def test_upsert_export_sender_new_and_none(store):
    assert store.upsert_export_sender(None, "Кто-то") is None
    sid = store.upsert_export_sender(456, "Дедушка")
    assert sid is not None
    row = store.conn.execute("SELECT display_name FROM senders WHERE id = ?", (sid,)).fetchone()
    assert row["display_name"] == "Дедушка"


def test_live_message_id_ignores_export_rows(store):
    add_message(store, msg_id=7, text="живое сообщение")
    store.insert_message(
        tg_message_id=8, tg_chat_id=-100, sender_id=None,
        ts=int(time.time()), kind="text", text="из экспорта", source="export",
    )
    assert store.live_message_id(-100, 7) is not None
    assert store.live_message_id(-100, 8) is None


def test_live_messages_at(store):
    ts = 1700000000
    row_id = add_message(store, msg_id=1, text="точное время", ts=ts)
    assert store.live_messages_at(-100, ts) == [(row_id, "точное время")]
    assert store.live_messages_at(-100, ts + 1) == []


def test_insert_transcript(store):
    row_id = add_message(store, msg_id=1, text=None, kind="voice")
    store.insert_transcript(message_id=row_id, text="привет", engine="fake")
    # idempotent re-insert (e.g. job retried after partial failure)
    store.insert_transcript(message_id=row_id, text="привет два", engine="fake")
    row = store.conn.execute(
        "SELECT text FROM transcripts WHERE message_id = ?", (row_id,)
    ).fetchone()
    assert row["text"] == "привет два"
