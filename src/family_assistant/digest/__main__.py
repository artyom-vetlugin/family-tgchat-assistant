"""CLI: python -m family_assistant.digest [--date YYYY-MM-DD] [--rebuild]

Default: catch up the wiki from the log.md watermark to yesterday (daily).
--rebuild: wipe Layer 2 and re-seed the whole history month by month.
--date:    (re)digest a single day, ignoring the watermark.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date

from ..config import get_settings
from ..store import Store
from ..wiki import Wiki
from .runner import run_digest


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Nightly wiki digest (M5)")
    parser.add_argument("--date", help="Digest a single day (YYYY-MM-DD), ignoring the watermark")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the wiki and re-seed all history month by month",
    )
    args = parser.parse_args()

    settings = get_settings()
    store = Store(settings.db_path)
    wiki = Wiki(settings.wiki_dir, settings.wiki_guide_path)
    only_date = date.fromisoformat(args.date) if args.date else None
    try:
        report = run_digest(
            store=store,
            settings=settings,
            wiki=wiki,
            rebuild=args.rebuild,
            only_date=only_date,
        )
    finally:
        store.close()
    print(report.render())


if __name__ == "__main__":
    main()
