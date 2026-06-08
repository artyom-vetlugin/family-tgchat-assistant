"""Pure command-helper tests (M7) — no dispatcher, no API."""

from datetime import date

from family_assistant import commands
from family_assistant.store.db import MessageRow


def _row(msg_id, text, ts=1700000000):
    return MessageRow(
        id=msg_id, tg_message_id=msg_id, tg_chat_id=-100, sender="Мама",
        ts=ts, reply_to=None, kind="text", text=text,
    )


# --- /spend ----------------------------------------------------------------


def test_format_spend_reply_empty():
    assert "пока ничего" in commands.format_spend_reply([], "2026-06")


def test_format_spend_reply_estimates_and_lists_models():
    rows = [
        {"model": "claude-haiku-4-5", "in_tokens": 1_000_000, "cached_tokens": 0,
         "cache_write_tokens": 0, "out_tokens": 0, "calls": 3},
    ]
    reply = commands.format_spend_reply(rows, "2026-06")
    assert "claude-haiku-4-5" in reply
    assert "3 вызовов" in reply
    assert "$1.00" in reply  # 1M input @ $1


def test_format_spend_reply_flags_unknown_model():
    rows = [{"model": "mystery", "in_tokens": 1_000_000, "cached_tokens": 0,
             "cache_write_tokens": 0, "out_tokens": 0, "calls": 1}]
    reply = commands.format_spend_reply(rows, "2026-06")
    assert "нет цены" in reply


# --- /find -----------------------------------------------------------------


def test_format_find_too_short():
    assert "минимум 3" in commands.format_find_reply([], "ab")


def test_format_find_empty_and_hits():
    assert "ничего не найдено" in commands.format_find_reply([], "поездка")
    reply = commands.format_find_reply([_row(1, "едем на дачу")], "дача")
    assert "едем на дачу" in reply


# --- /wiki -----------------------------------------------------------------


def test_resolve_wiki_target():
    pages = ["people/anna.md", "topics/dacha.md"]
    assert commands.resolve_wiki_target("", pages) is None
    assert commands.resolve_wiki_target("log", pages) == "log.md"
    assert commands.resolve_wiki_target("people/anna.md", pages) == "people/anna.md"
    assert commands.resolve_wiki_target("dacha", pages) == "topics/dacha.md"
    assert commands.resolve_wiki_target("несуществует", pages) is None


# --- /summary --------------------------------------------------------------


def test_resolve_summary_window_default_and_aliases():
    today = date(2026, 6, 8)
    label, start, end = commands.resolve_summary_window("", today)
    assert label == "за неделю"
    assert end == date(2026, 6, 7)  # yesterday
    assert start == date(2026, 6, 1)  # 7-day window ending yesterday
    assert commands.resolve_summary_window("день", today)[0] == "за день"
    assert commands.resolve_summary_window("month", today)[0] == "за месяц"


def test_parse_and_format_summary_from_log():
    log = (
        "# Журнал\n\n"
        "## 2026-06-02\n\nездили на дачу\n\n"
        "## 2026-06-05\n\nобсуждали школу\n\n"
        "## 2026-06-09\n\nвне окна\n\n"
    )
    text = commands.format_summary_from_log(log, "за неделю", date(2026, 6, 1), date(2026, 6, 7))
    assert "ездили на дачу" in text
    assert "обсуждали школу" in text
    assert "вне окна" not in text  # outside the window


def test_format_summary_none_when_empty_window():
    log = "# Журнал\n\n## 2026-06-02\n\n(нет сообщений)\n\n"
    assert commands.format_summary_from_log(log, "за неделю", date(2026, 6, 1), date(2026, 6, 7)) is None
    assert commands.format_summary_from_log(None, "за неделю", date(2026, 6, 1), date(2026, 6, 7)) is None
