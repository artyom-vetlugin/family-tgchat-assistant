"""Telegram bot: archives every group message; answers when addressed.

Requirements on the Telegram side:
- Privacy mode DISABLED for the bot (or make it a group admin), otherwise it
  only receives commands and mentions, not regular group messages.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from aiogram.types import Message

from .config import Settings
from .query import QueryEngine
from .store import Store
from .transcribe import TranscriptionWorker

log = logging.getLogger(__name__)

BOT_API_DOWNLOAD_LIMIT = 20 * 1024 * 1024  # 20MB

MEDIA_EXT = {"voice": ".oga", "video_note": ".mp4"}

router = Router()


def too_old_to_answer(msg_ts: int, now_ts: int, max_age_minutes: int) -> bool:
    """True for backlog @mentions delivered after the laptop was asleep —
    a late reply hours after the question is confusing (and costs a Sonnet
    call per queued mention). The message itself is archived regardless."""
    if max_age_minutes <= 0:
        return False
    return now_ts - msg_ts > max_age_minutes * 60


def archive_gap_warning(last_ts: int | None, now_ts: int) -> str | None:
    """Warn when the archive is older than Telegram's ~24h update retention:
    anything sent during the part of the gap beyond 24h is lost to the Bot API
    and only a fresh Desktop export can recover it."""
    if last_ts is None:
        return None
    gap_hours = (now_ts - last_ts) / 3600
    if gap_hours <= 24:
        return None
    return (
        f"archive gap of {gap_hours:.0f}h — Telegram only holds updates ~24h, "
        "messages beyond that are lost to the Bot API; re-run "
        "`python -m family_assistant.backfill` with a fresh export to recover them"
    )


def message_kind(message: Message) -> str:
    if message.text:
        return "text"
    if message.voice:
        return "voice"
    if message.video_note:
        return "video_note"
    if message.photo:
        return "photo"
    if message.video:
        return "video"
    if message.document:
        return "file"
    if message.sticker:
        return "sticker"
    return "other"


class BotApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = Store(settings.db_path)
        self.engine = QueryEngine(settings, self.store)
        self.bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=None),
        )
        self.dp = Dispatcher()
        self.dp.include_router(router)
        self.dp["app"] = self
        self.bot_username: str | None = None
        self.worker = TranscriptionWorker(
            settings.db_path, settings.media_dir, settings
        )

    def chat_allowed(self, chat_id: int) -> bool:
        allowed = self.settings.allowed_chat_ids
        return not allowed or chat_id in allowed

    def log_message(self, message: Message) -> int | None:
        """Persist any incoming message; returns the row id (None on duplicate)."""
        sender_id = None
        if message.from_user:
            sender_id = self.store.upsert_sender(
                message.from_user.id, message.from_user.full_name
            )
        text = message.text or message.caption
        row_id = self.store.insert_message(
            tg_message_id=message.message_id,
            tg_chat_id=message.chat.id,
            sender_id=sender_id,
            ts=int(message.date.timestamp()),
            kind=message_kind(message),
            text=text,
            reply_to=message.reply_to_message.message_id if message.reply_to_message else None,
        )
        if row_id and text:
            self.store.index_text(row_id, text)
        return row_id

    async def ingest_media(self, message: Message, row_id: int) -> None:
        """Download voice/video_note and enqueue a transcription job."""
        kind = message_kind(message)
        media_obj = message.voice if kind == "voice" else message.video_note
        if media_obj is None:
            return

        file_size = media_obj.file_size or 0
        if file_size > BOT_API_DOWNLOAD_LIMIT:
            self.store.insert_media(
                message_id=row_id, kind=kind, rel_path=None,
                bytes=file_size, skipped=True,
            )
            log.warning("media %s too big to download (%d bytes)", row_id, file_size)
            return

        when = message.date
        rel_path = f"live/{when:%Y/%m}/{message.message_id}{MEDIA_EXT[kind]}"
        abs_path = self.settings.media_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            await self.bot.download(media_obj.file_id, destination=abs_path)
        except Exception:
            log.exception("download failed for message %s", message.message_id)
            return

        self.store.insert_media(
            message_id=row_id, kind=kind, rel_path=rel_path,
            mime=getattr(media_obj, "mime_type", None), bytes=file_size,
        )
        job_id = self.store.create_job(job_type="transcribe", ref_id=row_id)
        self.worker.enqueue(job_id)

    def is_addressed_to_bot(self, message: Message) -> bool:
        """True when the bot is @mentioned or the message replies to the bot."""
        if message.reply_to_message and message.reply_to_message.from_user:
            if message.reply_to_message.from_user.username == self.bot_username:
                return True
        if message.text and self.bot_username:
            return f"@{self.bot_username}".lower() in message.text.lower()
        return False

    def strip_mention(self, text: str) -> str:
        if self.bot_username:
            text = text.replace(f"@{self.bot_username}", "")
        return text.strip()

    async def run(self) -> None:
        me = await self.bot.get_me()
        self.bot_username = me.username
        log.info("starting as @%s", self.bot_username)
        warning = archive_gap_warning(self.store.stats()["last_ts"], int(time.time()))
        if warning:
            log.warning(warning)
        self.worker.start()  # recovers unfinished transcription jobs
        try:
            await self.dp.start_polling(self.bot)
        finally:
            self.worker.stop()


@router.message(Command("id"))
async def cmd_id(message: Message) -> None:
    await message.reply(f"chat_id: {message.chat.id}")


@router.message(Command("stats"))
async def cmd_stats(message: Message, app: BotApp) -> None:
    if not app.chat_allowed(message.chat.id):
        return
    s = app.store.stats()
    reply = f"Сообщений в архиве: {s['n']}"
    if s["last_ts"]:
        # Last-message time makes a delivery gap visible right in the chat
        # (privacy mode off + count growing is the documented health check).
        last = datetime.fromtimestamp(s["last_ts"], tz=timezone.utc).astimezone()
        reply += f"\nПоследнее: {last:%Y-%m-%d %H:%M}"
    await message.reply(reply)


@router.message(F.chat.type.in_({"group", "supergroup"}))
async def on_group_message(message: Message, app: BotApp) -> None:
    if not app.chat_allowed(message.chat.id):
        log.warning("ignoring message from disallowed chat %s", message.chat.id)
        return

    row_id = app.log_message(message)

    if row_id and message_kind(message) in ("voice", "video_note"):
        await app.ingest_media(message, row_id)

    if message.text and app.is_addressed_to_bot(message):
        msg_ts = int(message.date.timestamp())
        if too_old_to_answer(msg_ts, int(time.time()), app.settings.answer_max_age_minutes):
            log.info(
                "skipping stale mention (%d min old): %.50s",
                (int(time.time()) - msg_ts) // 60, message.text,
            )
            return
        question = app.strip_mention(message.text)
        if not question:
            await message.reply("Я тут! Спросите меня что-нибудь 🙂")
            return
        asker = message.from_user.full_name if message.from_user else "кто-то"
        # If the question replies to a regular chat message (not to the bot),
        # give the agent that message id as the anchor for context lookup.
        reply_to = None
        replied = message.reply_to_message
        if replied and replied.from_user and replied.from_user.username != app.bot_username:
            reply_to = replied.message_id
        await app.bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = await app.engine.handle(message.chat.id, question, asker, reply_to=reply_to)
        except Exception:
            log.exception("query failed")
            answer = "Что-то пошло не так, попробуйте ещё раз чуть позже."
        await message.reply(answer)


@router.message(F.chat.type == "private")
async def on_private_message(message: Message, app: BotApp) -> None:
    """Private chat with a family member: treat every message as a question."""
    if message.text:
        asker = message.from_user.full_name if message.from_user else "кто-то"
        # Answer over the first allowed group chat's archive.
        allowed = app.settings.allowed_chat_ids
        chat_id = allowed[0] if allowed else message.chat.id
        await app.bot.send_chat_action(message.chat.id, "typing")
        try:
            answer = await app.engine.handle(chat_id, message.text, asker)
        except Exception:
            log.exception("query failed")
            answer = "Что-то пошло не так, попробуйте ещё раз чуть позже."
        await message.reply(answer)
