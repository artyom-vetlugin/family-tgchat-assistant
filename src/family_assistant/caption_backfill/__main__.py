"""Entrypoint: python -m family_assistant.caption_backfill [--max-batches N]

One-time (re-runnable, idempotent) captioning of archived photos via the
Batch API. Run after the M3 message backfill — it picks up every downloaded
photo that has no caption yet. Safe to kill: in-flight batches are resumed on
the next run. Use --max-batches 1 for a cost-capped trial (~500 photos).
"""

import argparse
import logging

from ..config import get_settings
from ..store import Store
from .runner import run_caption_backfill


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m family_assistant.caption_backfill",
        description="Caption archived photos with Haiku vision via the Batch API.",
    )
    ap.add_argument(
        "--max-batches", type=int, default=None,
        help="stop after submitting N batches (cost-capped trial run)",
    )
    ap.add_argument(
        "--retry-errored", action="store_true",
        help="reset previously-errored caption jobs and retry them "
             "(corrupt images simply re-skip)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    store = Store(settings.db_path)
    try:
        report = run_caption_backfill(
            store=store,
            media_dir=settings.media_dir,
            settings=settings,
            max_batches=args.max_batches,
            retry_errored=args.retry_errored,
        )
    finally:
        store.close()
    print(report.render())
    if report.pending_left:
        print(f"\nre-run without --max-batches to caption the remaining {report.pending_left}.")


if __name__ == "__main__":
    main()
