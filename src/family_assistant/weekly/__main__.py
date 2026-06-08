"""CLI: python -m family_assistant.weekly

Posts a once-a-week recap of the last 7 days to the chat (M7), assembled from
wiki/log.md ($0). Opt-in via WEEKLY_SUMMARY_ENABLED=true; a no-op (logs a skip)
when disabled or when the week has no digest entries. Launched by launchd on
Sunday evening (deploy/com.family.tgweekly.plist).
"""

from __future__ import annotations

import asyncio
import logging

from ..config import get_settings
from ..store import Store
from ..wiki import Wiki
from .runner import run_weekly


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    store = Store(settings.db_path)
    wiki = Wiki(settings.wiki_dir, settings.wiki_guide_path)
    try:
        report = asyncio.run(run_weekly(settings=settings, store=store, wiki=wiki))
    finally:
        store.close()
    print(report.render())


if __name__ == "__main__":
    main()
