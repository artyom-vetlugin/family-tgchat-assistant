"""Sonnet agentic retrieval loop over the chat archive.

Manual tool-use loop (not the SDK tool runner) so we can cap iterations and
keep full control over prompt-cache placement: tools + system prompt are
byte-stable and carry the cache breakpoint; the volatile question comes after.
"""

from __future__ import annotations

import logging
from datetime import datetime

from anthropic import AsyncAnthropic

from ..config import Settings
from ..embed import Embedder
from ..store.db import Store, format_rows
from ..wiki import Wiki

log = logging.getLogger(__name__)

# Frozen system prompt — do not interpolate anything volatile here (dates,
# names, ids); that would invalidate the prompt cache on every request.
AGENT_SYSTEM = """\
You are a helpful assistant living inside a family's Telegram group chat.
You have tools to search the family's chat archive (texts; later also voice
transcripts and photo descriptions).

Guidelines:
- Always answer in Russian, warmly and concisely.
- Use the tools to ground answers in the actual chat history. Re-run
  fts_search with different word forms (Russian morphology: "поехали",
  "поехать", "поездка") and synonyms if the first search misses.
- fts_search matches exact substrings — use it first for specific words,
  names and quotes. semantic_search matches by meaning, not wording — use it
  for paraphrased or conceptual questions ("когда говорили о здоровье" finds
  "болит голова") and whenever fts_search with different word forms still
  comes up empty.
- ALWAYS retrieve before answering or asking for clarification. If the
  question is vague or refers to "сообщение"/"голосовое"/"что он сказал"
  without specifics, call recent_window (24-48h) first — the person almost
  always means the most recent matching message. Voice messages appear with
  a [голосовое] prefix and their transcribed content.
- If the question was sent as a reply to a specific message, that message id
  is given in the user message — start with get_messages_around on it.
- When citing findings, mention who wrote it and when (date).
- For summary-type questions ("что обсуждали на прошлой неделе", "что нового у
  X") consult the wiki first: call list_wiki_index, then read_wiki_page on a
  relevant people/ or topics/ page, or read log.md for the dated journal. The
  wiki is a maintained digest of stable facts; fall back to fts_search /
  recent_window if it lacks the answer.
- Only ask a clarifying question if retrieval genuinely came up empty.
- The current date and the user's question follow in the user message.
"""

# Deterministic tool order — reordering invalidates the prompt cache.
TOOLS = [
    {
        "name": "fts_search",
        "description": (
            "Full-text substring search over the chat archive. Returns matching "
            "messages with sender, date and message id. The index matches "
            "substrings (trigrams), so prefer word stems: 'поезд' matches "
            "'поездка' and 'поездки'. Call this when looking for specific "
            "topics, words or events. Query must be at least 3 characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text (Russian), >=3 chars; prefer a word stem"},
                "limit": {"type": "integer", "description": "Max results, default 10"},
                "sender": {"type": "string", "description": "Filter by sender name (substring)"},
                "after": {"type": "string", "description": "Only messages on/after this date, ISO format YYYY-MM-DD"},
                "before": {"type": "string", "description": "Only messages on/before this date, ISO format YYYY-MM-DD"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_messages_around",
        "description": (
            "Fetch the conversation context around a specific message id "
            "(n messages before and after). Use after fts_search to read the "
            "surrounding discussion."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "integer", "description": "Telegram message id from a search result"},
                "n": {"type": "integer", "description": "Messages before/after, default 5"},
            },
            "required": ["message_id"],
        },
    },
    {
        "name": "recent_window",
        "description": (
            "Fetch the most recent chat messages verbatim. Use for summaries of "
            "a recent period ('что обсуждали вчера/на этой неделе')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "hours": {"type": "integer", "description": "How far back to look, in hours (e.g. 24, 168)"},
            },
            "required": ["hours"],
        },
    },
    {
        "name": "list_wiki_index",
        "description": (
            "List the wiki catalogue: the people/ and topics/ pages that the "
            "nightly digest maintains, each with a one-line summary. Start here "
            "for summary or 'what's new' questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_wiki_page",
        "description": (
            "Read a wiki page in full. path is 'people/<name>.md', "
            "'topics/<topic>.md', or 'log.md' (the dated digest journal — good "
            "for 'что обсуждали на прошлой неделе')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Wiki page path"}},
            "required": ["path"],
        },
    },
    {
        "name": "semantic_search",
        "description": (
            "Semantic (meaning-based) search over the chat archive using local "
            "embeddings. Unlike fts_search it matches paraphrases and concepts, "
            "not exact words: 'о здоровье' can surface 'болит голова'. Use it for "
            "fuzzy/conceptual questions or when fts_search keeps coming up empty. "
            "No minimum length."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (Russian); a phrase or concept works best"},
                "k": {"type": "integer", "description": "Number of results, default 10"},
                "sender": {"type": "string", "description": "Filter by sender name (substring)"},
                "after": {"type": "string", "description": "Only messages on/after this date, ISO format YYYY-MM-DD"},
                "before": {"type": "string", "description": "Only messages on/before this date, ISO format YYYY-MM-DD"},
            },
            "required": ["query"],
        },
    },
]


def _parse_date(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


class RetrievalAgent:
    def __init__(
        self,
        client: AsyncAnthropic,
        settings: Settings,
        store: Store,
        wiki: Wiki | None = None,
        embedder: Embedder | None = None,
    ):
        self.client = client
        self.settings = settings
        self.store = store
        self.wiki = wiki or Wiki(settings.wiki_dir, settings.wiki_guide_path)
        self.embedder = embedder or Embedder(settings)

    def _run_tool(self, name: str, tool_input: dict, chat_id: int) -> str:
        if name == "fts_search":
            rows = self.store.search(
                tool_input["query"],
                limit=int(tool_input.get("limit", 10)),
                sender=tool_input.get("sender"),
                after_ts=_parse_date(tool_input.get("after")),
                before_ts=_parse_date(tool_input.get("before")),
            )
            return format_rows(rows)
        if name == "get_messages_around":
            rows = self.store.get_messages_around(
                chat_id, int(tool_input["message_id"]), n=int(tool_input.get("n", 5))
            )
            return format_rows(rows)
        if name == "recent_window":
            rows = self.store.recent_window(chat_id, hours=int(tool_input["hours"]))
            return format_rows(rows)
        if name == "semantic_search":
            query_vec = self.embedder.encode_query(tool_input["query"])
            rows = self.store.knn(
                query_vec,
                model=self.embedder.model_name,
                k=int(tool_input.get("k", self.settings.semantic_search_k)),
                sender=tool_input.get("sender"),
                after_ts=_parse_date(tool_input.get("after")),
                before_ts=_parse_date(tool_input.get("before")),
            )
            return format_rows(rows)
        if name == "list_wiki_index":
            return self.wiki.read_index()
        if name == "read_wiki_page":
            try:
                content = self.wiki.read_page(tool_input["path"])
            except ValueError as exc:
                return f"Ошибка: {exc}"
            return content if content is not None else "(страница не существует)"
        return f"Unknown tool: {name}"

    async def answer(
        self, chat_id: int, question: str, asker: str, reply_to: int | None = None
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        context = f"Сейчас: {now}. Спрашивает {asker}."
        if reply_to is not None:
            context += f" Вопрос задан в ответ на сообщение {reply_to}."
        messages: list[dict] = [
            {"role": "user", "content": f"{context}\n\nВопрос: {question}"}
        ]

        for _ in range(self.settings.max_agent_iterations):
            response = await self.client.messages.create(
                model=self.settings.answer_model,
                max_tokens=2048,
                tools=TOOLS,
                system=[
                    {
                        "type": "text",
                        "text": AGENT_SYSTEM,
                        # Caches tools + system together (tools render first).
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            )
            self.store.record_spend(
                model=self.settings.answer_model, usage=response.usage
            )
            log.info(
                "agent turn: stop=%s in=%s cached=%s out=%s",
                response.stop_reason,
                response.usage.input_tokens,
                response.usage.cache_read_input_tokens,
                response.usage.output_tokens,
            )

            if response.stop_reason != "tool_use":
                return _final_text(response)

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = self._run_tool(block.name, block.input, chat_id)
                    except Exception as exc:  # tool bug must not kill the turn
                        log.exception("tool %s failed", block.name)
                        result = f"Ошибка инструмента: {exc}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )
            messages.append({"role": "user", "content": tool_results})

        # Iteration cap reached — ask for a final answer without tools.
        messages.append(
            {
                "role": "user",
                "content": "Заверши: дай финальный ответ на основе того, что уже нашлось.",
            }
        )
        response = await self.client.messages.create(
            model=self.settings.answer_model,
            max_tokens=2048,
            system=[
                {"type": "text", "text": AGENT_SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            tools=TOOLS,
            tool_choice={"type": "none"},
            messages=messages,
        )
        self.store.record_spend(model=self.settings.answer_model, usage=response.usage)
        return _final_text(response)

    async def answer_generic(self, question: str, asker: str) -> str:
        """Generic LLM question — no retrieval, single Sonnet call."""
        response = await self.client.messages.create(
            model=self.settings.answer_model,
            max_tokens=2048,
            system="Ты — дружелюбный ассистент в семейном телеграм-чате. Отвечай по-русски, кратко и по делу.",
            messages=[{"role": "user", "content": question}],
        )
        self.store.record_spend(model=self.settings.answer_model, usage=response.usage)
        return _final_text(response)


def _final_text(response) -> str:
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or "Не получилось сформулировать ответ, попробуйте ещё раз."
