"""Entrypoint: python -m family_assistant.backfill <export_dir> [--chat-id ID] [--transcribe]

One-time (re-runnable, idempotent) import of a Telegram Desktop export.
Transcription jobs are enqueued for the bot's worker to drain on next start;
pass --transcribe to run them inline here instead (multi-hour, resumable).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from ..config import get_settings
from ..store import Store
from .parser import export_chat_id
from .runner import run_backfill

log = logging.getLogger(__name__)


def derive_botapi_chat_id(export_id: int, chat_type: str) -> int:
    """Export stores the short positive chat id; the bot stores the Bot-API form."""
    if "supergroup" in chat_type or "channel" in chat_type:
        return int(f"-100{export_id}")
    return -export_id


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m family_assistant.backfill",
        description="Import a Telegram Desktop JSON export into the archive.",
    )
    ap.add_argument(
        "export_dir", type=Path, nargs="?", default=None,
        help="directory containing result.json (default: ./export)",
    )
    ap.add_argument(
        "--chat-id", type=int, default=None,
        help="Bot-API chat id to store under (default: ALLOWED_CHAT_IDS[0] from .env)",
    )
    ap.add_argument(
        "--transcribe", action="store_true",
        help="run pending transcription jobs inline after the import",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    export_dir = args.export_dir or settings.export_dir
    if not (export_dir / "result.json").is_file():
        sys.exit(f"error: no result.json in {export_dir}")

    chat_id = args.chat_id
    if chat_id is None:
        if not settings.allowed_chat_ids:
            sys.exit("error: pass --chat-id or set ALLOWED_CHAT_IDS in .env")
        chat_id = settings.allowed_chat_ids[0]

    # Sanity check: the export's own chat id should derive to the target id.
    result_json = json.loads((export_dir / "result.json").read_text(encoding="utf-8"))
    derived = derive_botapi_chat_id(export_chat_id(result_json), result_json.get("type", ""))
    if derived != chat_id:
        log.warning(
            "export chat id derives to %s but importing under %s — "
            "make sure this export is really the bot's chat (/id in the chat)",
            derived, chat_id,
        )

    store = Store(settings.db_path)
    try:
        report = run_backfill(
            export_dir=export_dir,
            store=store,
            media_dir=settings.media_dir,
            tg_chat_id=chat_id,
            settings=settings,
            transcribe_inline=args.transcribe,
        )
    finally:
        store.close()
    print(report.render())
    if not args.transcribe and report.transcribe_jobs:
        print(
            f"\n{report.transcribe_jobs} transcription jobs queued — start the bot "
            "to process them, or re-run with --transcribe."
        )


if __name__ == "__main__":
    main()
