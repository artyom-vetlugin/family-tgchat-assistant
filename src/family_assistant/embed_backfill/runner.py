"""One-time embedding backfill (M6): embed every archived message that has
searchable text but no vector yet.

Local + $0 (no API), so unlike the caption backfill there is no Batch API —
jobs are drained inline on one connection (the model loads once). Resumable:
re-runs only enqueue messages still missing an embedding, and the jobs
UNIQUE(job_type, ref_id) dedups, so killing and restarting never double-works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings
from ..embed import Embedder, EmbeddingWorker
from ..store import Store

log = logging.getLogger(__name__)


@dataclass
class EmbedBackfillReport:
    enqueued: int      # messages enqueued this run
    done: int          # embed jobs that finished ok
    errored: int       # embed jobs that exhausted retries
    chunks: int        # total embedding rows for the model (cumulative)
    pending_left: int  # embed jobs still pending (e.g. --limit cap)

    def render(self) -> str:
        return (
            "Embedding backfill:\n"
            f"  enqueued this run: {self.enqueued}\n"
            f"  embedded ok:       {self.done}\n"
            f"  errored:           {self.errored}\n"
            f"  vectors (chunks):  {self.chunks}\n"
            f"  pending left:      {self.pending_left}"
        )


def _count(store: Store, sql: str, params: tuple) -> int:
    return store.conn.execute(sql, params).fetchone()[0]


def run_embed_backfill(
    *,
    store: Store,
    settings: Settings,
    embedder: Embedder | None = None,
    limit: int | None = None,
    retry_errored: bool = False,
) -> EmbedBackfillReport:
    model = settings.embedding_model
    if retry_errored:
        reset = store.reset_errored_jobs("embed")
        if reset:
            log.info("reset %d errored embed jobs", reset)

    ids = store.messages_needing_embedding(model)
    if limit is not None:
        ids = ids[:limit]
    for message_id in ids:
        store.create_job(job_type="embed", ref_id=message_id)
    log.info("enqueued %d messages for embedding", len(ids))

    worker = EmbeddingWorker(settings.db_path, settings, embedder or Embedder(settings))
    pending = store.pending_jobs("embed")
    for i, job_id in enumerate(pending, 1):
        worker.process(store, job_id)
        if i % 200 == 0:
            log.info("embedded %d/%d", i, len(pending))

    return EmbedBackfillReport(
        enqueued=len(ids),
        done=_count(store, "SELECT COUNT(*) FROM jobs WHERE job_type='embed' AND state='done'", ()),
        errored=_count(store, "SELECT COUNT(*) FROM jobs WHERE job_type='embed' AND state='error'", ()),
        chunks=_count(store, "SELECT COUNT(*) FROM embeddings WHERE model=?", (model,)),
        pending_left=len(store.pending_jobs("embed")),
    )
