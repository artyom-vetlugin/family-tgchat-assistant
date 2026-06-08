"""Local semantic-search embeddings (M6).

Design mirrors transcribe.py / caption.py: one dedicated worker thread consuming
a queue of job ids, owning its own sqlite connection (WAL) and a lazily-loaded
sentence-transformers model. Embedding runs locally on CPU ($0 — no API call).

`embed` jobs use ref_id = messages.id (like transcribe, not caption). The text to
embed is the message's COALESCE(text, transcript, caption) — so a voice/photo
message is enqueued for embedding only after its transcript/caption lands.
"""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np

from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

_SENTINEL = -1


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of at most ~max_chars, breaking on whitespace.

    Short messages (the common case) come back as a single chunk; long voice
    transcripts split into several so their tail stays semantically searchable.
    A single word longer than max_chars is emitted whole rather than dropped.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for word in text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        chunks.append(current)
    return chunks


class Embedder:
    """Lazy wrapper around sentence-transformers. Loaded on first use; weights
    cache under models_dir. e5 needs `query: ` / `passage: ` prefixes and we
    L2-normalize so cosine similarity is a plain dot product."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model_name(self) -> str:
        return self.settings.embedding_model

    @property
    def dim(self) -> int:
        return self.settings.embedding_dim

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy, keep lazy

            log.info("loading embedding model %s ...", self.settings.embedding_model)
            self._model = SentenceTransformer(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
                cache_folder=str(self.settings.models_dir),
            )
            log.info("embedding model loaded")
        return self._model

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return model.encode(
            [f"passage: {t}" for t in texts],
            normalize_embeddings=True,
        ).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._load()
        return model.encode(
            f"query: {text}", normalize_embeddings=True
        ).astype(np.float32)


class EmbeddingWorker:
    def __init__(
        self,
        db_path: Path,
        settings: Settings,
        embedder: Embedder | None = None,
    ):
        self.db_path = db_path
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.queue: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="embed", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job_id: int) -> None:
        self.queue.put(job_id)

    def process(self, store: Store, job_id: int) -> None:
        """Run one job synchronously on the caller's connection (embed_backfill)."""
        self._process(store, job_id)

    # --- worker thread -------------------------------------------------------

    def _run(self) -> None:
        store = Store(self.db_path)  # this thread's own connection
        try:
            self.recover(store)
            while True:
                job_id = self.queue.get()
                if job_id == _SENTINEL:
                    return
                self._process(store, job_id)
        finally:
            store.close()

    def recover(self, store: Store) -> None:
        """Re-enqueue work left over from a previous run."""
        reset = store.reset_stale_jobs("embed")
        pending = store.pending_jobs("embed")
        if reset or pending:
            log.info("recovered %d stale + %d pending embed jobs", reset, len(pending))
        for job_id in pending:
            self.queue.put(job_id)

    def _process(self, store: Store, job_id: int) -> None:
        ref = store.job_ref(job_id)
        if ref is None:
            return
        _, message_id = ref
        store.claim_job(job_id)
        try:
            text = store.text_for_message(message_id)
            if text:
                chunks = chunk_text(text, self.settings.embedding_chunk_chars)
                if chunks:
                    vectors = self.embedder.encode_passages(chunks)
                    store.replace_embeddings(
                        message_id,
                        vectors,
                        model=self.embedder.model_name,
                        dim=self.embedder.dim,
                    )
            store.finish_job(job_id, ok=True, max_attempts=self.settings.job_max_attempts)
            log.info("embedded message %s", message_id)
        except Exception:
            log.exception("embed job %s failed", job_id)
            state = store.finish_job(
                job_id, ok=False, max_attempts=self.settings.job_max_attempts
            )
            if state == "pending":
                self.queue.put(job_id)  # retry
