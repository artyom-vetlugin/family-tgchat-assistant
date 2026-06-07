"""Tests for parsing Telegram Desktop export messages (pure functions)."""

from datetime import datetime

from family_assistant.backfill.parser import (
    classify_kind,
    export_chat_id,
    flatten_text,
    iter_messages,
    parse_from_id,
    parse_message,
    parse_ts,
)


def test_flatten_text_variants():
    assert flatten_text("привет") == "привет"
    assert flatten_text(None) == ""
    assert flatten_text([]) == ""
    assert (
        flatten_text(["смотри ", {"type": "link", "text": "https://example.com"}, " — интересно"])
        == "смотри https://example.com — интересно"
    )


def test_parse_from_id():
    assert parse_from_id("user123456") == 123456
    assert parse_from_id("channel987") is None
    assert parse_from_id(None) is None
    assert parse_from_id(42) == 42


def test_parse_ts_prefers_unixtime():
    assert parse_ts({"date_unixtime": "1700000000", "date": "2001-01-01T00:00:00"}) == 1700000000


def test_parse_ts_fallback_to_date():
    # Older exports lack date_unixtime; "date" is local naive time.
    expected = int(datetime.fromisoformat("2023-05-01T12:34:56").timestamp())
    assert parse_ts({"date": "2023-05-01T12:34:56"}) == expected


def test_classify_kind():
    assert classify_kind({"photo": "photos/p.jpg"}) == "photo"
    assert classify_kind({"file": "voice_messages/a.ogg", "media_type": "voice_message"}) == "voice"
    assert classify_kind({"file": "round_video_messages/v.mp4", "media_type": "video_message"}) == "video_note"
    assert classify_kind({"file": "video_files/v.mp4", "media_type": "video_file"}) == "video"
    assert classify_kind({"file": "stickers/s.webp", "media_type": "sticker"}) == "sticker"
    assert classify_kind({"file": "files/doc.pdf"}) == "file"
    assert classify_kind({"text": "просто текст"}) == "text"
    assert classify_kind({"text": ""}) == "other"


def test_parse_message_voice():
    pm = parse_message(
        {
            "id": 100,
            "type": "message",
            "date": "2023-05-01T12:34:56",
            "date_unixtime": "1682933696",
            "from": "Мама",
            "from_id": "user123456",
            "file": "voice_messages/audio_1@01-05-2023_12-34-56.ogg",
            "media_type": "voice_message",
            "mime_type": "audio/ogg",
            "text": "",
        }
    )
    assert pm.tg_message_id == 100
    assert pm.ts == 1682933696
    assert pm.from_id == 123456
    assert pm.kind == "voice"
    assert pm.media_rel_path == "voice_messages/audio_1@01-05-2023_12-34-56.ogg"
    assert pm.media_mime == "audio/ogg"
    assert not pm.media_not_included and not pm.is_service


def test_parse_message_file_not_included():
    pm = parse_message(
        {
            "id": 101,
            "type": "message",
            "date_unixtime": "1682933700",
            "from": "Папа",
            "from_id": "user654321",
            "file": "(File not included. Change data exporting settings to download.)",
            "media_type": "video_file",
            "text": "",
        }
    )
    assert pm.kind == "video"
    assert pm.media_rel_path is None
    assert pm.media_not_included


def test_parse_message_service_and_missing_from_id():
    pm = parse_message(
        {
            "id": 102,
            "type": "service",
            "date_unixtime": "1682933800",
            "actor": "Мама",
            "actor_id": "user123456",
            "action": "pin_message",
            "text": "",
        }
    )
    assert pm.is_service
    assert pm.from_id is None  # no from_id on service messages


def test_iter_messages_and_chat_id():
    data = {
        "name": "Семья",
        "type": "private_supergroup",
        "id": 123456789,
        "messages": [
            {"id": 1, "type": "message", "date_unixtime": "1", "from": "А", "from_id": "user1", "text": "привет"},
            {"id": 2, "type": "message", "date_unixtime": "2", "from": "Б", "from_id": "user2", "text": "пока"},
        ],
    }
    assert export_chat_id(data) == 123456789
    assert [pm.tg_message_id for pm in iter_messages(data)] == [1, 2]
