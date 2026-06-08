"""Entrypoint: python -m family_assistant.video_backfill [--limit N] [--retry-errored]

One-time (re-runnable, idempotent) processing of archived videos (M7): whisper
transcript + keyframe captions, indexed for search. Export videos are local so
size is no issue. Transcription is $0/local; keyframe captions are live Haiku
(video_keyframes per video — default 1). Use --limit N for a cost-capped trial;
safe to kill and re-run to resume.
"""

import argparse
import logging

from ..config import get_settings
from ..store import Store
from .runner import run_video_backfill


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m family_assistant.video_backfill",
        description="Transcribe + caption archived videos for search.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="process at most N videos this run (trial / incremental)",
    )
    ap.add_argument(
        "--retry-errored", action="store_true",
        help="reset previously-errored video jobs and retry them",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    store = Store(settings.db_path)
    try:
        report = run_video_backfill(
            store=store,
            settings=settings,
            limit=args.limit,
            retry_errored=args.retry_errored,
        )
    finally:
        store.close()
    print(report.render())
    if report.pending_left:
        print(f"\nre-run to process the remaining {report.pending_left}.")


if __name__ == "__main__":
    main()
