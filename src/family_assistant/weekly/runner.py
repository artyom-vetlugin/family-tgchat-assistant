"""Weekly chat summary (M7): a Sunday-evening recap posted back to the chat.

Opt-in (weekly_summary_enabled). Reuses the nightly digest's output: it stitches
the last 7 days of dated entries from wiki/log.md ($0 — the digest already wrote
them) rather than making a fresh LLM pass. Runs as a standalone launchd job, so
it builds its own aiogram Bot just to send one message (a send-only Bot does not
conflict with the polling bot).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

from .. import commands
from ..config import Settings
from ..digest.runner import _resolve_chat_id
from ..store import Store
from ..wiki import Wiki

log = logging.getLogger(__name__)


@dataclass
class WeeklyReport:
    enabled: bool = True
    chat_id: int | None = None
    sent: bool = False
    note: str = ""

    def render(self) -> str:
        return (
            "=== Weekly summary ===\n"
            f"enabled: {self.enabled}\n"
            f"chat_id: {self.chat_id}\n"
            f"sent:    {self.sent}\n"
            f"note:    {self.note}"
        )


def build_weekly_text(wiki: Wiki, today: date) -> str | None:
    """Stitch the last 7 days (ending yesterday) of log.md, or None if empty."""
    end = today - timedelta(days=1)
    start = end - timedelta(days=6)
    return commands.format_summary_from_log(wiki.read_page("log.md"), "за неделю", start, end)


async def run_weekly(
    *,
    settings: Settings,
    store: Store,
    wiki: Wiki | None = None,
    bot=None,  # aiogram Bot; injected in tests, built lazily otherwise
    today: date | None = None,
) -> WeeklyReport:
    if not settings.weekly_summary_enabled:
        log.info("weekly summary disabled (WEEKLY_SUMMARY_ENABLED=false)")
        return WeeklyReport(enabled=False, note="disabled")
    if wiki is None:
        wiki = Wiki(settings.wiki_dir, settings.wiki_guide_path)
    if today is None:
        today = date.today()

    chat_id = _resolve_chat_id(store, settings)
    report = WeeklyReport(chat_id=chat_id)
    if chat_id is None:
        report.note = "no chat_id (set DIGEST_CHAT_ID or ALLOWED_CHAT_IDS)"
        log.warning(report.note)
        return report

    text = build_weekly_text(wiki, today)
    if not text:
        report.note = "no log entries for the past week — nothing to post"
        log.info(report.note)
        return report

    created = bot is None
    if created:
        from aiogram import Bot
        from aiogram.client.default import DefaultBotProperties

        bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=None),
        )
    try:
        await bot.send_message(chat_id, text)
        report.sent = True
        log.info("weekly summary posted to %s", chat_id)
    finally:
        if created:
            await bot.session.close()
    return report
