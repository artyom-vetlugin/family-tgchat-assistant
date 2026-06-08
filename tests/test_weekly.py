"""Weekly summary tests (M7) — fake Bot, no Telegram."""

import asyncio
from datetime import date

import pytest

from family_assistant.store import Store
from family_assistant.weekly.runner import build_weekly_text, run_weekly
from family_assistant.wiki import Wiki


class FakeSettings:
    weekly_summary_enabled = True
    digest_chat_id = -100
    allowed_chat_ids = [-100]
    telegram_bot_token = "x"
    wiki_dir = None
    wiki_guide_path = None


class FakeBot:
    def __init__(self):
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "test.db")
    wiki = Wiki(tmp_path / "wiki")
    yield store, wiki
    store.close()


def seed_log(wiki):
    wiki.append_log(date(2026, 6, 2), "ездили на дачу")
    wiki.append_log(date(2026, 6, 5), "обсуждали школу")


def test_build_weekly_text_window(env):
    _, wiki = env
    seed_log(wiki)
    text = build_weekly_text(wiki, date(2026, 6, 8))  # week = Jun 1..7
    assert "ездили на дачу" in text
    assert "обсуждали школу" in text


def test_build_weekly_text_none_when_empty(env):
    _, wiki = env
    text = build_weekly_text(wiki, date(2026, 6, 8))
    assert text is None


def test_run_weekly_sends_when_enabled(env):
    store, wiki = env
    seed_log(wiki)
    bot = FakeBot()
    report = asyncio.run(
        run_weekly(settings=FakeSettings(), store=store, wiki=wiki, bot=bot, today=date(2026, 6, 8))
    )
    assert report.sent is True
    assert len(bot.sent) == 1
    assert bot.sent[0][0] == -100


def test_run_weekly_skips_when_disabled(env):
    store, wiki = env
    seed_log(wiki)
    settings = FakeSettings()
    settings.weekly_summary_enabled = False
    bot = FakeBot()
    report = asyncio.run(
        run_weekly(settings=settings, store=store, wiki=wiki, bot=bot, today=date(2026, 6, 8))
    )
    assert report.enabled is False
    assert report.sent is False
    assert bot.sent == []


def test_run_weekly_skips_when_no_entries(env):
    store, wiki = env  # empty log
    bot = FakeBot()
    report = asyncio.run(
        run_weekly(settings=FakeSettings(), store=store, wiki=wiki, bot=bot, today=date(2026, 6, 8))
    )
    assert report.sent is False
    assert bot.sent == []
    assert "nothing to post" in report.note
