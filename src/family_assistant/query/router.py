"""Intent router: one cheap Haiku call decides whether a question needs
chat-history retrieval or is a generic LLM question."""

from __future__ import annotations

import json
import logging

from anthropic import AsyncAnthropic

log = logging.getLogger(__name__)

INTENTS = ("search_history", "summarize", "context_qa", "generic_llm")

# Frozen prompt — keep byte-stable so it stays cache/KV friendly.
ROUTER_SYSTEM = """\
You classify questions sent to a family Telegram chat assistant bot.
The chat archive contains the family's messages, voice transcripts and photo descriptions.

Categories:
- search_history: asks to find specific messages/photos/moments in the chat ("когда мы...", "найди...", "что Маша писала про...")
- summarize: asks for a summary of a period or topic ("что обсуждали на прошлой неделе")
- context_qa: answerable from chat context ("какой у нас план на лето?", "что решили насчёт школы?")
- generic_llm: general knowledge, no chat context needed ("сколько варить яйцо", "переведи текст")

Respond with the category only.
"""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"intent": {"type": "string", "enum": list(INTENTS)}},
    "required": ["intent"],
    "additionalProperties": False,
}

# Unambiguous Russian openings that don't need the Haiku router (M7) — skipping
# it saves a call (and its spend). Keep conservative: only heads where the
# intent is obvious. Everything else falls through to classify_intent.
_SEARCH_PREFIXES = ("найди", "найти", "поищи", "ищи", "покажи ")
_SUMMARY_PREFIXES = ("что обсуждали", "о чём говорили", "о чем говорили",
                     "подведи итог", "сделай summary", "итоги ")


def shortcut_intent(question: str) -> str | None:
    """Cheap keyword shortcut around the Haiku router; None = ask the model."""
    q = question.strip().lower()
    if q.startswith(_SEARCH_PREFIXES):
        return "search_history"
    if q.startswith(_SUMMARY_PREFIXES):
        return "summarize"
    return None


async def classify_intent(
    client: AsyncAnthropic, model: str, question: str, store=None
) -> str:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=64,
            system=ROUTER_SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": ROUTER_SCHEMA}},
            messages=[{"role": "user", "content": question}],
        )
        if store is not None:
            store.record_spend(model=model, usage=response.usage)
        text = next(b.text for b in response.content if b.type == "text")
        intent = json.loads(text)["intent"]
        if intent in INTENTS:
            return intent
    except Exception:
        log.exception("intent routing failed, falling back to context_qa")
    # When in doubt, retrieval-backed answering is the safe default.
    return "context_qa"
