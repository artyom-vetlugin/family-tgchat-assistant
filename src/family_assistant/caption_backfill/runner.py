"""Photo caption backfill via the Anthropic Batch API (M4, 50% off).

Resumable like the M3 backfill: every submitted job carries its batch id
(jobs.batch_id), so a killed run picks up in-flight batches on restart instead
of re-submitting (and re-paying for) them. A completed run is a no-op.

Retry semantics piggyback on the jobs table: an errored batch result sends the
job back to 'pending' (until job_max_attempts), and the outer loop re-submits
pending jobs in a fresh batch — so one invocation normally drains everything.

FTS idempotency: a batch result is applied only while its job is still
inflight on that batch — re-collecting after a mid-collection crash never
double-indexes a caption.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..caption import caption_system_blocks, encode_image, image_content_block, response_text
from ..config import Settings
from ..store import Store

log = logging.getLogger(__name__)


@dataclass
class CaptionBackfillReport:
    photos_uncaptioned: int = 0  # photo media rows lacking a caption at start
    jobs_created: int = 0
    batches: int = 0
    submitted: int = 0
    succeeded: int = 0
    errored: int = 0
    skipped_encode: int = 0  # corrupt/unreadable images, no API call made
    indexed: int = 0
    pending_left: int = 0  # only when --max-batches cut the run short
    errored_retried: int = 0  # errored jobs revived by --retry-errored

    def render(self) -> str:
        return (
            "=== Caption backfill report ===\n"
            f"photos without caption: {self.photos_uncaptioned}\n"
            f"  errored jobs retried: {self.errored_retried}\n"
            f"  caption jobs created: {self.jobs_created}\n"
            f"batches submitted:      {self.batches}\n"
            f"  requests submitted:   {self.submitted}\n"
            f"  succeeded:            {self.succeeded}\n"
            f"  errored:              {self.errored}\n"
            f"  skipped (bad image):  {self.skipped_encode}\n"
            f"captions indexed:       {self.indexed}\n"
            f"jobs left pending:      {self.pending_left}"
        )


def run_caption_backfill(
    *,
    store: Store,
    media_dir: Path,
    settings: Settings,
    client=None,  # anthropic.Anthropic; injected in tests, built lazily otherwise
    max_batches: int | None = None,
    retry_errored: bool = False,
) -> CaptionBackfillReport:
    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    report = CaptionBackfillReport()

    # Inflight jobs without a batch id are stale (crash between claim and
    # submit); jobs WITH a batch id are resumed below, not reset.
    store.reset_stale_jobs("caption", unbatched_only=True)

    # Revive terminally-errored jobs first, so they re-enter the pending pool
    # below (transient API failures retry; corrupt images simply re-skip).
    if retry_errored:
        report.errored_retried = store.reset_errored_jobs("caption")
        log.info("retrying %d previously-errored caption jobs", report.errored_retried)

    media_ids = store.photo_media_without_caption()
    report.photos_uncaptioned = len(media_ids)
    for media_id in media_ids:
        job_id = store.create_job(job_type="caption", ref_id=media_id)
        state = store.job_state(job_id)
        if state and state[0] == "pending":
            report.jobs_created += 1  # new or retried; UNIQUE makes re-runs cheap

    while True:
        # Collect everything in flight first — including batches a previous
        # (killed) run submitted.
        for batch_id in store.batch_ids_for_jobs("caption"):
            _collect_batch(client, store, settings, batch_id, report)

        pending = store.pending_jobs("caption")
        if not pending:
            break
        if max_batches is not None and report.batches >= max_batches:
            report.pending_left = len(pending)
            log.info("--max-batches reached, %d jobs left pending", len(pending))
            break

        submitted_any = False
        chunk_size = settings.caption_batch_chunk
        for i in range(0, len(pending), chunk_size):
            if max_batches is not None and report.batches >= max_batches:
                break
            chunk = pending[i : i + chunk_size]
            if _submit_chunk(client, store, media_dir, settings, chunk, report):
                submitted_any = True
        if not submitted_any:
            break  # nothing submittable (e.g. every image failed to encode)

    return report


def _submit_chunk(
    client, store: Store, media_dir: Path, settings: Settings,
    job_ids: list[int], report: CaptionBackfillReport,
) -> bool:
    """Build and submit one batch; returns True if a batch was created."""
    requests = []
    claimed: list[int] = []
    for job_id in job_ids:
        ref = store.job_ref(job_id)
        if ref is None:
            continue
        _, media_id = ref
        media = store.media_row(media_id)
        store.claim_job(job_id)
        if media is None or media[1] is None:
            store.finish_job(job_id, ok=False, max_attempts=1)  # nothing to retry
            report.skipped_encode += 1
            continue
        try:
            b64 = encode_image(
                media_dir / media[1],
                long_edge=settings.caption_resize_long_edge,
                quality=settings.caption_jpeg_quality,
            )
        except Exception:
            log.warning("unreadable image for media %s (%s), skipping", media_id, media[1])
            store.finish_job(job_id, ok=False, max_attempts=1)  # corrupt won't heal
            report.skipped_encode += 1
            continue
        requests.append(
            {
                "custom_id": f"caption-{job_id}",
                "params": {
                    "model": settings.caption_model,
                    "max_tokens": settings.caption_max_tokens,
                    "system": caption_system_blocks(),
                    "messages": [
                        {"role": "user", "content": [image_content_block(b64)]}
                    ],
                },
            }
        )
        claimed.append(job_id)

    if not requests:
        return False
    batch = client.messages.batches.create(requests=requests)
    for job_id in claimed:
        store.set_job_batch(job_id, batch.id)
    report.batches += 1
    report.submitted += len(requests)
    log.info("submitted batch %s with %d caption requests", batch.id, len(requests))
    return True


def _collect_batch(
    client, store: Store, settings: Settings, batch_id: str,
    report: CaptionBackfillReport,
) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = getattr(batch, "request_counts", None)
        log.info("batch %s: %s (%s)", batch_id, batch.processing_status, counts)
        time.sleep(settings.caption_poll_seconds)

    for result in client.messages.batches.results(batch_id):
        job_id = int(result.custom_id.rsplit("-", 1)[1])
        state = store.job_state(job_id)
        # Apply only to jobs still inflight on THIS batch: a re-collected
        # (already-applied) result must not double-index, and a job that was
        # re-submitted elsewhere is no longer this batch's to finish.
        if state != ("inflight", batch_id):
            continue
        ref = store.job_ref(job_id)
        if ref is None:
            continue
        _, media_id = ref
        if result.result.type == "succeeded":
            text = response_text(result.result.message.content)
            if text:
                store.insert_caption(
                    media_id=media_id, text=text, model=settings.caption_model
                )
                media = store.media_row(media_id)
                if media:
                    store.index_text(media[0], text)
                    report.indexed += 1
            store.finish_job(job_id, ok=True, max_attempts=settings.job_max_attempts)
            report.succeeded += 1
        else:  # errored | expired | canceled — retried until attempts run out
            log.warning("batch result %s for job %s", result.result.type, job_id)
            store.finish_job(job_id, ok=False, max_attempts=settings.job_max_attempts)
            report.errored += 1
