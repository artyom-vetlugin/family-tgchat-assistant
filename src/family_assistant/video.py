"""Video understanding (M7): spoken content + keyframe descriptions.

Combines the M2 transcription path and the M4 captioning path into one worker:
each video produces one whisper transcript plus N keyframe captions (default 1),
stitched into a single searchable string stored in `transcripts` and indexed for
FTS + semantic search. Same thread+queue+WAL shape as TranscriptionWorker.

`video` jobs use ref_id = messages.id (like transcribe; the worker looks up the
media row internally). The combined text lands via insert_transcript/index_text
— no per-frame media rows, so the existing `transcripts`/`captions` schema is
untouched. faster-whisper reads the .mp4 audio directly via PyAV; keyframes are
decoded with PyAV too (already a transitive dependency, no new package).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from .caption import Captioner
from .config import Settings
from .store import Store
from .transcribe import Transcriber

log = logging.getLogger(__name__)

_SENTINEL = -1


def extract_keyframes(path: Path, n: int) -> list:
    """Decode up to `n` keyframes spread across the video's duration as PIL
    images. Best-effort: a short/corrupt video may yield fewer (or zero) frames
    — the transcript alone still makes the message searchable."""
    if n <= 0:
        return []
    import av  # heavy, transitive via faster-whisper — keep lazy

    frames: list = []
    try:
        container = av.open(str(path))
    except Exception:
        log.warning("could not open video for keyframes: %s", path)
        return []
    try:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            return []
        duration = container.duration  # in av.time_base (microseconds), may be None
        fractions = [(i + 1) / (n + 1) for i in range(n)]
        for frac in fractions:
            try:
                if duration:
                    container.seek(int(duration * frac), backward=True, any_frame=False)
                frame = next(container.decode(stream), None)
                if frame is not None:
                    frames.append(frame.to_image())
            except Exception:
                log.warning("keyframe decode failed at %.2f for %s", frac, path)
                continue
    finally:
        container.close()
    return frames


def build_video_text(transcript: str, captions: list[str]) -> str:
    """Stitch transcript + keyframe captions into one searchable body. The
    `[видео]` prefix is added by MessageRow.format (kind-based), not here — so
    the stored text stays prefix-free, like voice transcripts."""
    parts: list[str] = []
    if transcript:
        parts.append(transcript)
    caps = [c.strip() for c in captions if c and c.strip()]
    if caps:
        parts.append("в кадре: " + "; ".join(caps))
    return " | ".join(parts)


class VideoWorker:
    def __init__(
        self,
        db_path: Path,
        media_dir: Path,
        settings: Settings,
        transcriber: Transcriber | None = None,
        captioner: Captioner | None = None,
        embed_enqueue: Callable[[int], None] | None = None,
    ):
        self.db_path = db_path
        self.media_dir = media_dir
        self.settings = settings
        # Share the bot's whisper Transcriber (avoid a 2nd ~1.5GB model load;
        # CTranslate2 inference is thread-safe). The Captioner is NOT shared —
        # its `last_usage` is per-call state that two threads would race on.
        self.transcriber = transcriber or Transcriber(settings)
        self.captioner = captioner or Captioner(settings)
        self.embed_enqueue = embed_enqueue
        self.queue: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="video", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job_id: int) -> None:
        self.queue.put(job_id)

    def process(self, store: Store, job_id: int) -> None:
        """Run one job synchronously on the caller's connection (backfill CLI)."""
        self._process(store, job_id)

    def _enqueue_embed(self, store: Store, message_id: int) -> None:
        if self.embed_enqueue is None:
            return
        job_id = store.create_job(job_type="embed", ref_id=message_id)
        self.embed_enqueue(job_id)

    # --- worker thread -------------------------------------------------------

    def _run(self) -> None:
        store = Store(self.db_path)
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
        reset = store.reset_stale_jobs("video")
        pending = store.pending_jobs("video")
        if reset or pending:
            log.info("recovered %d stale + %d pending video jobs", reset, len(pending))
        for job_id in pending:
            self.queue.put(job_id)

    def _caption_frames(self, store: Store, frames: list) -> list[str]:
        captions: list[str] = []
        for img in frames:
            text = self.captioner.caption_pil(img)
            usage = getattr(self.captioner, "last_usage", None)
            if usage is not None:
                store.record_spend(model=self.captioner.model, usage=usage)
            if text:
                captions.append(text)
        return captions

    def _process(self, store: Store, job_id: int) -> None:
        ref = store.job_ref(job_id)
        if ref is None:
            return
        _, message_id = ref
        store.claim_job(job_id)
        try:
            media = store.media_for_message(message_id)
            if media is None or media[0] is None or media[1]:  # missing or skipped
                raise FileNotFoundError(f"no downloadable video for message {message_id}")
            path = self.media_dir / media[0]
            transcript = self.transcriber.transcribe(path)
            frames = extract_keyframes(path, self.settings.video_keyframes)
            captions = self._caption_frames(store, frames)
            combined = build_video_text(transcript, captions)
            if combined:
                store.insert_transcript(
                    message_id=message_id, text=combined,
                    engine=self.transcriber.engine, lang=self.settings.whisper_lang,
                )
                store.index_text(message_id, combined)
                self._enqueue_embed(store, message_id)
            store.finish_job(job_id, ok=True, max_attempts=self.settings.job_max_attempts)
            log.info("processed video message %s (%d chars, %d frames)",
                     message_id, len(combined), len(captions))
        except Exception:
            log.exception("video job %s failed", job_id)
            state = store.finish_job(
                job_id, ok=False, max_attempts=self.settings.job_max_attempts
            )
            if state == "pending":
                self.queue.put(job_id)  # retry
