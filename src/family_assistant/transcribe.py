"""Local transcription of voice messages and video circles (M2).

Design: one dedicated worker thread consuming a queue of job ids. The thread
owns its own sqlite connection (sqlite3 connections are thread-bound; WAL mode
makes a second connection safe) and a lazily-loaded faster-whisper model.
CTranslate2 releases the GIL during inference, so the bot stays responsive.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

_SENTINEL = -1


class Transcriber:
    """Lazy wrapper around faster-whisper. Loaded on first use (~1.5GB RAM,
    first run downloads weights to models_dir)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def engine(self) -> str:
        return f"faster-whisper:{self.settings.whisper_model}"

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, keep lazy

            log.info("loading whisper model %s ...", self.settings.whisper_model)
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
                download_root=str(self.settings.models_dir),
            )
            log.info("whisper model loaded")
        return self._model

    def transcribe(self, path: Path) -> str:
        model = self._load()
        segments, _info = model.transcribe(
            str(path), language=self.settings.whisper_lang, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


class TranscriptionWorker:
    def __init__(
        self,
        db_path: Path,
        media_dir: Path,
        settings: Settings,
        transcriber: Transcriber | None = None,
        embed_enqueue: Callable[[int], None] | None = None,
    ):
        self.db_path = db_path
        self.media_dir = media_dir
        self.settings = settings
        self.transcriber = transcriber or Transcriber(settings)
        # Hook to enqueue a follow-on embed job once the transcript is indexed
        # (M6); None when running standalone (e.g. backfill --transcribe).
        self.embed_enqueue = embed_enqueue
        self.queue: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="transcribe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job_id: int) -> None:
        self.queue.put(job_id)

    def _enqueue_embed(self, store: Store, message_id: int) -> None:
        """Queue an embed job for the freshly-transcribed message (M6)."""
        if self.embed_enqueue is None:
            return
        job_id = store.create_job(job_type="embed", ref_id=message_id)
        self.embed_enqueue(job_id)

    def process(self, store: Store, job_id: int) -> None:
        """Run one job synchronously on the caller's connection (backfill --transcribe)."""
        self._process(store, job_id)

    # --- worker thread -------------------------------------------------------

    def _run(self) -> None:
        store = Store(self.db_path)  # this thread's own connection
        try:
            self.recover(store)
            while True:
                job_id = self.queue.get()
                if job_id == _SENTINEL:
                    return
                self._process(store, job_id)
        finally:
            store.close()

    def recover(self, store: Store) -> None:
        """Re-enqueue work left over from a previous run."""
        reset = store.reset_stale_jobs("transcribe")
        pending = store.pending_jobs("transcribe")
        if reset or pending:
            log.info("recovered %d stale + %d pending transcribe jobs", reset, len(pending))
        for job_id in pending:
            self.queue.put(job_id)

    def _process(self, store: Store, job_id: int) -> None:
        ref = store.job_ref(job_id)
        if ref is None:
            return
        _, message_id = ref
        store.claim_job(job_id)
        try:
            media = store.media_for_message(message_id)
            if media is None or media[0] is None:
                raise FileNotFoundError(f"no media file recorded for message {message_id}")
            path = self.media_dir / media[0]
            text = self.transcriber.transcribe(path)
            if text:
                store.insert_transcript(
                    message_id=message_id, text=text, engine=self.transcriber.engine,
                    lang=self.settings.whisper_lang,
                )
                store.index_text(message_id, text)
                self._enqueue_embed(store, message_id)
            state = store.finish_job(job_id, ok=True, max_attempts=self.settings.job_max_attempts)
            log.info("transcribed message %s (%d chars)", message_id, len(text))
        except Exception:
            log.exception("transcription job %s failed", job_id)
            state = store.finish_job(
                job_id, ok=False, max_attempts=self.settings.job_max_attempts
            )
            if state == "pending":
                self.queue.put(job_id)  # retry
