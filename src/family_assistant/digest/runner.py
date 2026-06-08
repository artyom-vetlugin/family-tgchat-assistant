"""Nightly digest: one Haiku pass per period maintains the LLM Wiki (M5).

Mirrors the RetrievalAgent shape but synchronous (this is a CLI/background job,
never on the event loop) and driven by Haiku, not Sonnet. Prompt-cache layout
per ROADMAP: system = wiki_guide (frozen) → start-of-period index snapshot
(cache breakpoint here) → the period's messages in the user turn. Cache hits
land on turns 2..N within a period (verify via cache_read_input_tokens).

Two modes:
- daily catch-up (default): digest each day from the log.md watermark+1 to
  yesterday. A fresh wiki with no watermark digests only yesterday — run with
  --rebuild to seed history.
- rebuild seed: wipe Layer 2, then one Haiku pass per calendar month
  oldest→newest so each month builds on the pages the prior months produced.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..config import Settings
from ..store.db import Store, format_rows
from ..wiki import Wiki

log = logging.getLogger(__name__)

SAVED = "Сохранено."

DIGEST_SYSTEM_INTRO = (
    "Ты ведёшь вики семейного телеграм-чата по ночному дайджесту. "
    "Ниже — правила ведения вики, затем текущий индекс страниц. Следуй правилам.\n\n"
)

DIGEST_TOOLS = [
    {
        "name": "list_wiki_index",
        "description": "Показать актуальный каталог страниц вики (people/ и topics/).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_wiki_page",
        "description": (
            "Прочитать страницу вики целиком перед её дополнением. "
            "path вида 'people/anna.md' или 'topics/dacha.md'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Путь к странице"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_wiki_page",
        "description": (
            "Создать или перезаписать страницу вики целиком (передавай полный "
            "новый текст — дополняй прочитанное, не теряй факты). "
            "Можно писать только people/<имя>.md и topics/<тема>.md."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к странице"},
                "content": {"type": "string", "description": "Полный текст страницы (markdown, по-русски)"},
            },
            "required": ["path", "content"],
        },
    },
]


@dataclass
class DigestReport:
    mode: str = "daily"
    chat_id: int | None = None
    periods_processed: int = 0  # periods that hit the API
    periods_empty: int = 0
    pages_written: int = 0
    api_turns: int = 0
    note: str = ""

    def render(self) -> str:
        lines = [
            "=== Digest report ===",
            f"mode:              {self.mode}",
            f"chat_id:           {self.chat_id}",
            f"periods processed: {self.periods_processed}",
            f"periods empty:     {self.periods_empty}",
            f"pages written:     {self.pages_written}",
            f"api turns:         {self.api_turns}",
        ]
        if self.note:
            lines.append(f"note:              {self.note}")
        return "\n".join(lines)


def _day_start_ts(d: date) -> int:
    return int(datetime(d.year, d.month, d.day).timestamp())


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _next_month(d: date) -> date:
    return date(d.year + 1, 1, 1) if d.month == 12 else date(d.year, d.month + 1, 1)


def _final_text(response) -> str:
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or "(дайджест не сформулирован)"


class DigestAgent:
    """Haiku tool-loop that maintains wiki pages for one period of messages."""

    def __init__(self, client, settings: Settings, wiki: Wiki):
        self.client = client
        self.settings = settings
        self.wiki = wiki

    def _system_blocks(self) -> list[dict]:
        # guide (frozen) + start-of-period index snapshot; cache covers both.
        prefix = DIGEST_SYSTEM_INTRO + self.wiki.guide()
        return [
            {"type": "text", "text": prefix},
            {
                "type": "text",
                "text": "# Текущий индекс\n\n" + self.wiki.read_index(),
                "cache_control": {"type": "ephemeral"},
            },
        ]

    def _run_tool(self, name: str, tool_input: dict) -> str:
        if name == "list_wiki_index":
            return self.wiki.regenerate_index()
        if name == "read_wiki_page":
            content = self.wiki.read_page(tool_input["path"])
            return content if content is not None else "(страница не существует)"
        if name == "write_wiki_page":
            try:
                self.wiki.write_page(tool_input["path"], tool_input["content"])
            except (ValueError, KeyError) as exc:
                return f"Ошибка: {exc}"
            return SAVED
        return f"Неизвестный инструмент: {name}"

    def digest_period(self, label: date, messages_text: str) -> tuple[str, int, int]:
        """Run the loop for one period; returns (summary, pages_written, turns)."""
        system = self._system_blocks()  # snapshot the index once for the whole run
        user = (
            f"Период до {label.isoformat()} включительно. Ниже все сообщения чата "
            "за этот период.\nОбнови вики по правилам: при необходимости дополни "
            "страницы people/ и topics/ через инструменты (сначала прочитай "
            "страницу, чтобы не потерять факты). Затем дай краткое résumé (2–5 "
            "предложений) для журнала — это твой финальный ответ без инструментов.\n\n"
            f"{messages_text}"
        )
        messages: list[dict] = [{"role": "user", "content": user}]
        pages_written = 0
        turns = 0

        for _ in range(self.settings.max_digest_iterations):
            response = self.client.messages.create(
                model=self.settings.digest_model,
                max_tokens=self.settings.digest_max_tokens,
                tools=DIGEST_TOOLS,
                system=system,
                messages=messages,
            )
            turns += 1
            log.info(
                "digest turn: stop=%s in=%s cached=%s out=%s",
                response.stop_reason,
                response.usage.input_tokens,
                response.usage.cache_read_input_tokens,
                response.usage.output_tokens,
            )
            if response.stop_reason != "tool_use":
                return _final_text(response), pages_written, turns

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    result = self._run_tool(block.name, block.input)
                except Exception as exc:  # a tool bug must not kill the digest
                    log.exception("digest tool %s failed", block.name)
                    result = f"Ошибка инструмента: {exc}"
                if block.name == "write_wiki_page" and result == SAVED:
                    pages_written += 1
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            messages.append({"role": "user", "content": results})

        # Iteration cap — force a final journal entry without tools.
        messages.append(
            {"role": "user", "content": "Заверши: дай краткое résumé периода для журнала."}
        )
        response = self.client.messages.create(
            model=self.settings.digest_model,
            max_tokens=self.settings.digest_max_tokens,
            tools=DIGEST_TOOLS,
            tool_choice={"type": "none"},
            system=system,
            messages=messages,
        )
        turns += 1
        return _final_text(response), pages_written, turns


def _resolve_chat_id(store: Store, settings: Settings) -> int | None:
    if settings.digest_chat_id is not None:
        return settings.digest_chat_id
    if settings.allowed_chat_ids:
        return settings.allowed_chat_ids[0]
    return None


def _periods(mode: str, start: date, end: date):
    """Yield (window_start, window_end_inclusive) per period; label = window_end."""
    if mode == "monthly":
        cur = max(_month_start(start), start)
        while cur <= end:
            nxt = _next_month(cur)
            yield (cur, min(nxt - timedelta(days=1), end))
            cur = nxt
    else:  # daily
        cur = start
        while cur <= end:
            yield (cur, cur)
            cur += timedelta(days=1)


def run_digest(
    *,
    store: Store,
    settings: Settings,
    wiki: Wiki | None = None,
    client=None,  # anthropic.Anthropic; injected in tests, built lazily otherwise
    today: date | None = None,
    rebuild: bool = False,
    only_date: date | None = None,
) -> DigestReport:
    if wiki is None:
        wiki = Wiki(settings.wiki_dir, settings.wiki_guide_path)
    if today is None:
        today = date.today()
    yesterday = today - timedelta(days=1)

    chat_id = _resolve_chat_id(store, settings)
    report = DigestReport(chat_id=chat_id)
    if chat_id is None:
        report.note = "no chat_id (set DIGEST_CHAT_ID or ALLOWED_CHAT_IDS)"
        log.warning(report.note)
        return report

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    if rebuild:
        wiki.reset()
        first_ts = store.first_message_date(chat_id)
        if first_ts is None:
            report.note = "no messages to seed from"
            return report
        start = datetime.fromtimestamp(first_ts).date()
        report.mode = "monthly"
    elif only_date is not None:
        start = end = only_date
        report.mode = "daily"
    else:
        watermark = wiki.last_digested_date()
        start = watermark + timedelta(days=1) if watermark else yesterday
        report.mode = "daily"

    if only_date is None:
        end = yesterday
    if start > end:
        report.note = "nothing to digest (already up to date)"
        return report

    agent = DigestAgent(client, settings, wiki)
    for win_start, win_end in _periods(report.mode, start, end):
        rows = store.messages_between(
            chat_id, _day_start_ts(win_start), _day_start_ts(win_end + timedelta(days=1))
        )
        if not rows:
            wiki.append_log(win_end, "(нет сообщений)")
            report.periods_empty += 1
            continue
        summary, pages, turns = agent.digest_period(win_end, format_rows(rows))
        wiki.append_log(win_end, summary)
        wiki.regenerate_index()
        report.periods_processed += 1
        report.pages_written += pages
        report.api_turns += turns
        log.info("digested %s..%s: %d pages, %d turns", win_start, win_end, pages, turns)

    wiki.regenerate_index()
    return report
