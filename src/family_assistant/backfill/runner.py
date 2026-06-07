"""Backfill orchestration: result.json → SQLite archive (source='export').

Dedup happens at insert time, not insert-then-delete: the query paths
(search / recent_window / get_messages_around) read messages without a source
filter, so a live+export pair for one logical message would surface as a
duplicate everywhere. A live row always wins; the export may still contribute
its media file (live rows logged before M2 have no media row).

FTS idempotency: index_text is a plain FTS insert, so it runs only when
insert_message actually created a row — re-running backfill never double-indexes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..store import Store
from .media import copy_media
from .parser import ParsedMessage, iter_messages

log = logging.getLogger(__name__)

TRANSCRIBABLE_KINDS = ("voice", "video_note")
PROGRESS_EVERY = 10  # inline-transcription progress log interval


def norm_text(text: str) -> str:
    """Normalization for the fallback dedup match (whitespace/case-insensitive)."""
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class BackfillReport:
    total: int = 0
    imported: int = 0
    deduped_live: int = 0      # live row with the same tg_message_id existed
    deduped_hash: int = 0      # live row matched on (ts, normalized text)
    already_export: int = 0    # export row from a previous backfill run
    service_skipped: int = 0
    media_copied: int = 0
    media_reused: int = 0      # destination already had identical content
    media_upgraded: int = 0    # placeholder row replaced by a real file (fuller export)
    media_missing: int = 0     # path in result.json but file absent on disk
    media_skipped_notincluded: int = 0
    transcribe_jobs: int = 0
    transcribed_inline: int = 0
    errors: int = 0

    def render(self) -> str:
        return (
            "=== Backfill report ===\n"
            f"messages in export:   {self.total}\n"
            f"  imported:           {self.imported}\n"
            f"  deduped (live id):  {self.deduped_live}\n"
            f"  deduped (ts+text):  {self.deduped_hash}\n"
            f"  already imported:   {self.already_export}\n"
            f"  service skipped:    {self.service_skipped}\n"
            f"  errors:             {self.errors}\n"
            f"media copied:         {self.media_copied}\n"
            f"  reused (re-run):    {self.media_reused}\n"
            f"  upgraded (re-run):  {self.media_upgraded}\n"
            f"  missing on disk:    {self.media_missing}\n"
            f"  not in export:      {self.media_skipped_notincluded}\n"
            f"transcribe jobs:      {self.transcribe_jobs}\n"
            f"  transcribed inline: {self.transcribed_inline}"
        )


def run_backfill(
    *,
    export_dir: Path,
    store: Store,
    media_dir: Path,
    tg_chat_id: int,
    settings: Settings,
    transcribe_inline: bool = False,
    worker=None,  # TranscriptionWorker; injected in tests, built lazily otherwise
) -> BackfillReport:
    result_json = json.loads((export_dir / "result.json").read_text(encoding="utf-8"))
    report = BackfillReport()

    for pm in iter_messages(result_json):
        report.total += 1
        try:
            _process_message(
                pm, store=store, export_dir=export_dir, media_dir=media_dir,
                tg_chat_id=tg_chat_id, report=report,
            )
        except Exception:
            log.exception("failed to import export message id=%s", pm.tg_message_id)
            report.errors += 1

    log.info(
        "import done: %d imported, %d deduped, %d already present",
        report.imported, report.deduped_live + report.deduped_hash, report.already_export,
    )

    if transcribe_inline:
        _drain_transcribe_jobs(store, media_dir, settings, report, worker)
    return report


def _process_message(
    pm: ParsedMessage,
    *,
    store: Store,
    export_dir: Path,
    media_dir: Path,
    tg_chat_id: int,
    report: BackfillReport,
) -> None:
    if pm.is_service:
        report.service_skipped += 1
        return

    # Dedup: a live row wins; the export contributes media if the live row has none.
    live_id = store.live_message_id(tg_chat_id, pm.tg_message_id)
    if live_id is None and pm.text:
        normalized = norm_text(pm.text)
        for cand_id, cand_text in store.live_messages_at(tg_chat_id, pm.ts):
            if cand_text and norm_text(cand_text) == normalized:
                live_id = cand_id
                report.deduped_hash += 1
                break
        else:
            live_id = None
    elif live_id is not None:
        report.deduped_live += 1

    if live_id is not None:
        if pm.media_rel_path and _media_attachable(store, live_id):
            _attach_media(pm, row_id=live_id, store=store, export_dir=export_dir,
                          media_dir=media_dir, report=report)
        return

    sender_id = store.upsert_export_sender(pm.from_id, pm.from_name)
    row_id = store.insert_message(
        tg_message_id=pm.tg_message_id,
        tg_chat_id=tg_chat_id,
        sender_id=sender_id,
        ts=pm.ts,
        kind=pm.kind,
        text=pm.text or None,
        reply_to=pm.reply_to,
        source="export",
    )
    if row_id is None:  # previous backfill run inserted it
        report.already_export += 1
        row_id = store.export_message_id(tg_chat_id, pm.tg_message_id)
        if row_id is None:
            return
    else:
        report.imported += 1
        if pm.text:
            store.index_text(row_id, pm.text)

    if pm.media_not_included and not store.has_media(row_id):
        store.insert_media(
            message_id=row_id, kind=pm.kind, rel_path=None,
            mime=pm.media_mime, skipped=True,
        )
        report.media_skipped_notincluded += 1
    elif pm.media_rel_path and _media_attachable(store, row_id):
        _attach_media(pm, row_id=row_id, store=store, export_dir=export_dir,
                      media_dir=media_dir, report=report)


def _media_attachable(store: Store, row_id: int) -> bool:
    """True when the row has no media yet or only a placeholder (rel_path NULL —
    sentinel from a partial export, or a >20MB live download skip) that a fuller
    export can now upgrade. A recorded real file is never replaced."""
    existing = store.media_for_message(row_id)
    return existing is None or existing[0] is None


def _attach_media(
    pm: ParsedMessage,
    *,
    row_id: int,
    store: Store,
    export_dir: Path,
    media_dir: Path,
    report: BackfillReport,
) -> None:
    """Copy the export file, record/upgrade the media row, enqueue transcription."""
    copied = copy_media(
        export_dir=export_dir, media_dir=media_dir,
        src_rel=pm.media_rel_path, msg_id=pm.tg_message_id, ts=pm.ts,
    )
    if copied is None:
        report.media_missing += 1
        return
    if store.has_media(row_id):  # placeholder row from a partial export / live skip
        store.upgrade_media(
            row_id, rel_path=copied.rel_path,
            mime=pm.media_mime, bytes=copied.bytes, sha256=copied.sha256,
        )
        report.media_upgraded += 1
    else:
        store.insert_media(
            message_id=row_id, kind=pm.kind, rel_path=copied.rel_path,
            mime=pm.media_mime, bytes=copied.bytes, sha256=copied.sha256,
        )
        report.media_reused += copied.reused
        report.media_copied += not copied.reused
    if pm.kind in TRANSCRIBABLE_KINDS:
        store.create_job(job_type="transcribe", ref_id=row_id)
        report.transcribe_jobs += 1


def _drain_transcribe_jobs(
    store: Store, media_dir: Path, settings: Settings, report: BackfillReport, worker
) -> None:
    """Synchronously run all pending transcribe jobs (multi-hour, resumable).

    Loops over pending_jobs() rather than just-created ids: failed jobs go back
    to 'pending' (the worker's queue-based retry isn't consumed here) and reach
    'error' after max_attempts, so the loop terminates.
    """
    if worker is None:
        from ..transcribe import TranscriptionWorker

        worker = TranscriptionWorker(settings.db_path, media_dir, settings)
    store.reset_stale_jobs("transcribe")
    total = len(store.pending_jobs("transcribe"))
    while pending := store.pending_jobs("transcribe"):
        for job_id in pending:
            worker.process(store, job_id)
            report.transcribed_inline += 1
            if report.transcribed_inline % PROGRESS_EVERY == 0:
                log.info("inline transcription: %d/%d jobs", report.transcribed_inline, total)
