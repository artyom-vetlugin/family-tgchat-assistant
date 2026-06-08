"""One-time video backfill (M7): transcribe + caption archived videos.

Export videos (M3) land as `media` rows with a local file but no job — this
enqueues a `video` job per such message and drains them inline on one connection
(whisper loads once; keyframe captions are live Haiku, not Batch — with
video_keyframes=1 that's ~1 vision call per video, pennies; whisper dominates
wall-clock at $0). Resumable: a message that already has a transcript is skipped,
so killing and re-running never double-works.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import Settings
from ..store import Store
from ..video import VideoWorker

log = logging.getLogger(__name__)


@dataclass
class VideoBackfillReport:
    videos_without_transcript: int  # at start
    enqueued: int                   # jobs enqueued this run
    done: int                       # video jobs that finished ok
    errored: int                    # video jobs that exhausted retries
    pending_left: int               # still pending (e.g. --limit cap)

    def render(self) -> str:
        return (
            "Video backfill:\n"
            f"  videos without transcript: {self.videos_without_transcript}\n"
            f"  enqueued this run:         {self.enqueued}\n"
            f"  processed ok:              {self.done}\n"
            f"  errored:                   {self.errored}\n"
            f"  pending left:              {self.pending_left}"
        )


def _count(store: Store, state: str) -> int:
    return store.conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE job_type='video' AND state=?", (state,)
    ).fetchone()[0]


def run_video_backfill(
    *,
    store: Store,
    settings: Settings,
    worker: VideoWorker | None = None,
    limit: int | None = None,
    retry_errored: bool = False,
) -> VideoBackfillReport:
    if retry_errored:
        reset = store.reset_errored_jobs("video")
        if reset:
            log.info("reset %d errored video jobs", reset)
    # Stale inflight from a previous killed run → back to pending.
    store.reset_stale_jobs("video")

    message_ids = store.video_media_without_transcript()
    total = len(message_ids)
    if limit is not None:
        message_ids = message_ids[:limit]
    for message_id in message_ids:
        store.create_job(job_type="video", ref_id=message_id)
    log.info("enqueued %d videos for processing", len(message_ids))

    worker = worker or VideoWorker(settings.db_path, settings.media_dir, settings)
    pending = store.pending_jobs("video")
    for i, job_id in enumerate(pending, 1):
        worker.process(store, job_id)
        log.info("processed %d/%d video jobs", i, len(pending))

    return VideoBackfillReport(
        videos_without_transcript=total,
        enqueued=len(message_ids),
        done=_count(store, "done"),
        errored=_count(store, "error"),
        pending_left=len(store.pending_jobs("video")),
    )
