"""Pure parsing of a Telegram Desktop export (result.json) — no DB, no I/O.

Format notes (defensive: written before the real export exists):
- top level: {"name", "type", "id": <short positive chat id>, "messages": [...]}
- message "text" is a plain string OR a list mixing strings and entity dicts
  like {"type": "link", "text": "..."}.
- "date_unixtime" (string of unix seconds) exists in newer exports; older ones
  only have "date" — a *local naive* ISO timestamp on the exporting machine.
  We assume export and bot run on the same machine (same tz), matching how the
  bot stores int(message.date.timestamp()).
- media lives in "photo" or "file" (path relative to the export dir) with
  "media_type" ∈ {voice_message, video_message, video_file, sticker,
  animation, audio_file}; if media wasn't exported the field holds a literal
  "(File not included. ...)" sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

# Telegram Desktop writes this literal string in place of a path when the
# export settings excluded the file (size cap / media types unchecked).
FILE_NOT_INCLUDED_PREFIX = "(File not included."

_MEDIA_TYPE_TO_KIND = {
    "voice_message": "voice",
    "video_message": "video_note",
    "video_file": "video",
    "sticker": "sticker",
    "animation": "sticker",
    "audio_file": "file",
}


@dataclass
class ParsedMessage:
    tg_message_id: int
    ts: int
    from_id: int | None
    from_name: str | None
    reply_to: int | None
    kind: str
    text: str
    media_rel_path: str | None  # relative to the export dir
    media_mime: str | None
    media_not_included: bool
    is_service: bool


def export_chat_id(result_json: dict) -> int:
    """Top-level chat id — the short positive form (no -100 prefix)."""
    return int(result_json["id"])


def flatten_text(raw: str | list | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in raw)


def parse_from_id(raw: str | int | None) -> int | None:
    """'user123456' -> 123456; channels/anonymous/missing -> None."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    if raw.startswith("user"):
        try:
            return int(raw[len("user"):])
        except ValueError:
            return None
    return None


def parse_ts(msg: dict) -> int:
    raw = msg.get("date_unixtime")
    if raw:
        return int(raw)
    return int(datetime.fromisoformat(msg["date"]).timestamp())


def classify_kind(msg: dict) -> str:
    if msg.get("photo") is not None:
        return "photo"
    media_type = msg.get("media_type")
    if media_type in _MEDIA_TYPE_TO_KIND:
        return _MEDIA_TYPE_TO_KIND[media_type]
    if msg.get("file") is not None:
        return "file"
    if flatten_text(msg.get("text")):
        return "text"
    return "other"


def _media_field(msg: dict) -> tuple[str | None, bool]:
    """(rel_path, not_included) for the message's media file, if any."""
    raw = msg.get("photo") or msg.get("file")
    if raw is None:
        return None, False
    if isinstance(raw, str) and raw.startswith(FILE_NOT_INCLUDED_PREFIX):
        return None, True
    return raw, False


def parse_message(msg: dict) -> ParsedMessage:
    media_rel_path, not_included = _media_field(msg)
    return ParsedMessage(
        tg_message_id=int(msg["id"]),
        ts=parse_ts(msg),
        from_id=parse_from_id(msg.get("from_id")),
        from_name=msg.get("from") or msg.get("actor"),
        reply_to=msg.get("reply_to_message_id"),
        kind=classify_kind(msg),
        text=flatten_text(msg.get("text")),
        media_rel_path=media_rel_path,
        media_mime=msg.get("mime_type"),
        media_not_included=not_included,
        is_service=msg.get("type") == "service",
    )


def iter_messages(result_json: dict) -> Iterator[ParsedMessage]:
    for msg in result_json.get("messages", []):
        yield parse_message(msg)
