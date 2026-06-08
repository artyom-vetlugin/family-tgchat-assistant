"""Pure helpers behind the bot's slash commands (M7).

The aiogram handlers in bot.py are thin wrappers; the formatting/parsing logic
lives here so it can be unit-tested without a dispatcher. All user-facing text
is Russian.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from . import pricing
from .store.db import MessageRow, format_rows

TELEGRAM_LIMIT = 4096

_LOG_DATE_RE = re.compile(r"^## (\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)

# /summary period aliases → window length in days, and a Russian label.
_PERIODS: dict[str, tuple[int, str]] = {
    "день": (1, "за день"),
    "day": (1, "за день"),
    "вчера": (1, "за день"),
    "неделя": (7, "за неделю"),
    "неделю": (7, "за неделю"),
    "week": (7, "за неделю"),
    "месяц": (30, "за месяц"),
    "month": (30, "за месяц"),
}


def _truncate(text: str) -> str:
    if len(text) <= TELEGRAM_LIMIT:
        return text
    return text[: TELEGRAM_LIMIT - 1].rstrip() + "…"


# --- /spend ----------------------------------------------------------------


def format_spend_reply(rows: list[dict], month: str) -> str:
    """Render `store.spend_summary` rows as a Russian /spend report with an
    approximate USD estimate (prices change — the reply says «приблизительно»)."""
    if not rows:
        return f"Расходы {month}: пока ничего."
    total_usd, unknown = pricing.estimate_usd(rows)
    lines = [f"Расходы за {month} (приблизительно):"]
    for r in rows:
        tokens = (
            r.get("in_tokens", 0)
            + r.get("cached_tokens", 0)
            + r.get("cache_write_tokens", 0)
            + r.get("out_tokens", 0)
        )
        lines.append(
            f"• {r['model']}: {r.get('calls', 0)} вызовов, "
            f"{tokens // 1000}k токенов"
        )
    lines.append(f"Итого ≈ ${total_usd:.2f}")
    if unknown:
        lines.append("⚠️ нет цены для: " + ", ".join(unknown))
    return _truncate("\n".join(lines))


# --- /find -----------------------------------------------------------------


def format_find_reply(rows: list[MessageRow], query: str) -> str:
    if len(query.strip()) < 3:
        return "Запрос слишком короткий — нужно минимум 3 символа."
    if not rows:
        return f"По запросу «{query}» ничего не найдено."
    return _truncate(f"Нашёл по «{query}»:\n{format_rows(rows)}")


# --- /wiki -----------------------------------------------------------------


def resolve_wiki_target(arg: str, pages: list[str]) -> str | None:
    """Map a /wiki argument to an existing page path, or None to show the index.

    `pages` is the wiki page list (e.g. ['people/anna.md', 'topics/dacha.md']).
    A path-like arg is normalized; a bare word matches a page by substring.
    """
    arg = arg.strip()
    if not arg:
        return None
    if arg in ("log", "log.md"):
        return "log.md"
    candidate = arg if arg.endswith(".md") else None
    if candidate and candidate in pages:
        return candidate
    needle = arg.lower().removesuffix(".md")
    for page in pages:
        if needle in page.lower():
            return page
    return None


# --- /summary --------------------------------------------------------------


def resolve_summary_window(arg: str, today: date) -> tuple[str, date, date]:
    """(label, start_date, end_date_inclusive) for a /summary period arg.

    Defaults to the last week. end is yesterday (the digest only writes a day's
    log entry after the day is over), start is end - (days-1)."""
    days, label = _PERIODS.get(arg.strip().lower(), (7, "за неделю"))
    end = today - timedelta(days=1)
    start = end - timedelta(days=days - 1)
    return label, start, end


def parse_log_entries(log_text: str) -> list[tuple[date, str]]:
    """Parse log.md into (date, body) entries in document order."""
    entries: list[tuple[date, str]] = []
    matches = list(_LOG_DATE_RE.finditer(log_text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(log_text)
        body = log_text[m.end():end].strip()
        entries.append((date.fromisoformat(m.group(1)), body))
    return entries


def format_summary_from_log(
    log_text: str | None, label: str, start: date, end: date
) -> str | None:
    """Stitch the log.md entries within [start, end] into a Russian summary, or
    None when the window has no entries (caller falls back to the agent)."""
    if not log_text:
        return None
    window = [
        (d, body)
        for d, body in parse_log_entries(log_text)
        if start <= d <= end and body and body != "(нет сообщений)"
    ]
    if not window:
        return None
    head = f"Итоги {label} ({start:%d.%m}–{end:%d.%m}):"
    parts = [head] + [f"\n📅 {d:%d.%m}\n{body}" for d, body in window]
    return _truncate("\n".join(parts))
