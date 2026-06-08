"""Router shortcut tests (M7) — the keyword bypass around the Haiku call."""

import asyncio
from types import SimpleNamespace

from family_assistant.query import engine as engine_mod
from family_assistant.query.router import shortcut_intent


def test_shortcut_search_prefixes():
    assert shortcut_intent("найди фото с дачи") == "search_history"
    assert shortcut_intent("Найти сообщение про школу") == "search_history"
    assert shortcut_intent("  поищи рецепт  ") == "search_history"


def test_shortcut_summary_prefixes():
    assert shortcut_intent("что обсуждали на прошлой неделе") == "summarize"
    assert shortcut_intent("подведи итог дня") == "summarize"


def test_shortcut_none_for_ambiguous():
    assert shortcut_intent("сколько варить яйцо") is None
    assert shortcut_intent("какой план на лето?") is None


def test_engine_uses_shortcut_without_calling_router(monkeypatch):
    """A shortcut hit must skip classify_intent entirely (saves the Haiku call)."""
    called = {"router": False, "answer": None}

    async def fake_classify(*args, **kwargs):
        called["router"] = True
        return "generic_llm"

    monkeypatch.setattr(engine_mod, "classify_intent", fake_classify)

    eng = engine_mod.QueryEngine.__new__(engine_mod.QueryEngine)
    eng.settings = SimpleNamespace(router_model="claude-haiku-4-5")
    eng.store = None
    eng.client = None

    async def fake_answer(chat_id, question, asker, reply_to=None):
        called["answer"] = question
        return "ok"

    eng.agent = SimpleNamespace(answer=fake_answer, answer_generic=None)

    result = asyncio.run(eng.handle(-100, "найди фото", "Мама"))
    assert result == "ok"
    assert called["router"] is False  # shortcut bypassed the router
    assert called["answer"] == "найди фото"
