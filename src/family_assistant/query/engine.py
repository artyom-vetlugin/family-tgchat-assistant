"""Query engine facade: route the intent, then answer."""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic

from ..config import Settings
from ..embed import Embedder
from ..store.db import Store
from .agent import RetrievalAgent
from .router import classify_intent, shortcut_intent

log = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, settings: Settings, store: Store, embedder: Embedder | None = None):
        self.settings = settings
        self.store = store
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.agent = RetrievalAgent(self.client, settings, store, embedder=embedder)

    async def handle(
        self, chat_id: int, question: str, asker: str, reply_to: int | None = None
    ) -> str:
        intent = shortcut_intent(question) or await classify_intent(
            self.client, self.settings.router_model, question, store=self.store
        )
        log.info("intent=%s question=%r reply_to=%s", intent, question[:80], reply_to)
        if intent == "generic_llm":
            return await self.agent.answer_generic(question, asker)
        return await self.agent.answer(chat_id, question, asker, reply_to=reply_to)
