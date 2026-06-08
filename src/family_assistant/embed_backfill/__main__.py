"""Entrypoint: python -m family_assistant.embed_backfill [--limit N] [--retry-errored]

One-time (re-runnable, idempotent) embedding of every archived message that has
searchable text — message text, voice transcripts and photo captions — for
semantic search (M6). Runs locally on CPU at $0. The first run downloads the
embedding model to models/ and embeds the whole archive; use --limit N for a
quick trial. Safe to kill: re-run to resume where it stopped.
"""

import argparse
import logging

from ..config import get_settings
from ..store import Store
from .runner import run_embed_backfill


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m family_assistant.embed_backfill",
        description="Embed archived messages locally for semantic search.",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="embed at most N messages this run (trial / incremental)",
    )
    ap.add_argument(
        "--retry-errored", action="store_true",
        help="reset previously-errored embed jobs and retry them",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    store = Store(settings.db_path)
    try:
        report = run_embed_backfill(
            store=store,
            settings=settings,
            limit=args.limit,
            retry_errored=args.retry_errored,
        )
    finally:
        store.close()
    print(report.render())
    if report.pending_left:
        print(f"\nre-run to embed the remaining {report.pending_left}.")


if __name__ == "__main__":
    main()
