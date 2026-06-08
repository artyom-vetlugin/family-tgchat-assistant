"""Image captioning via Haiku vision (M4).

Design mirrors transcribe.py: one dedicated worker thread consuming a queue of
job ids, with its own sqlite connection (WAL). Live photos get an immediate
single API call (1-2/day — the 50% batch discount is pennies and not worth a
scheduler); the export backfill uses the Batch API (caption_backfill package).

Caption jobs use ref_id = media.id (transcribe jobs use messages.id).
"""

from __future__ import annotations

import base64
import io
import logging
import queue
import threading
from collections.abc import Callable
from pathlib import Path

from .config import Settings
from .store import Store

log = logging.getLogger(__name__)

_SENTINEL = -1

# Frozen prompt (byte-stable for prompt caching — never interpolate anything
# volatile). Below Haiku's minimum cacheable prefix, so it silently won't
# cache today; the breakpoint is still correct and future-proof.
CAPTION_PROMPT = (
    "Опиши фотографию из семейного чата для поискового архива. "
    "1–3 предложения, по делу: что изображено, кто или что в кадре "
    "(люди, предметы, животные, место, время года), и весь видимый текст "
    "дословно. Без вступлений, без догадок, без оценок. Отвечай по-русски."
)

IMAGE_MIME = "image/jpeg"


def encode_pil_image(img, *, long_edge: int = 1024, quality: int = 80) -> str:
    """Resize a PIL image to <=long_edge px and return a base64 JPEG string.

    Shared by the photo path (encode_image, from a file) and the video path
    (M7, from an in-memory keyframe) — neither touches an on-disk original, so
    Layer 1 stays immutable. Resizing only shrinks what goes over the wire."""
    img = img.convert("RGB")  # PNG/WebP/GIF (first frame) / video frame → JPEG-able
    img.thumbnail((long_edge, long_edge))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def encode_image(path: Path, *, long_edge: int = 1024, quality: int = 80) -> str:
    """Resize to <=long_edge px and re-encode as JPEG; return base64 string.

    The original file in media/ is never touched (Layer 1 is immutable) —
    resizing only shrinks what goes over the wire (image tokens cost money).
    Raises on unreadable/corrupt files.
    """
    from PIL import Image  # local import keeps bot startup lean

    with Image.open(path) as img:
        return encode_pil_image(img, long_edge=long_edge, quality=quality)


def image_content_block(b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": IMAGE_MIME, "data": b64},
    }


def caption_system_blocks() -> list[dict]:
    """Shared between the live Captioner and the Batch API backfill."""
    return [
        {
            "type": "text",
            "text": CAPTION_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def response_text(content) -> str:
    """Concatenate text blocks of a Messages API response content list."""
    return "".join(b.text for b in content if b.type == "text").strip()


class Captioner:
    """Lazy sync Anthropic client; one vision call per photo (live path).

    Sync on purpose: it runs on the worker thread, never on the event loop.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None
        # usage of the most recent call — workers read it to record spend (M7).
        self.last_usage = None

    @property
    def model(self) -> str:
        return self.settings.caption_model

    def _load(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def _caption_b64(self, b64: str) -> str:
        response = self._load().messages.create(
            model=self.settings.caption_model,
            max_tokens=self.settings.caption_max_tokens,
            system=caption_system_blocks(),
            messages=[{"role": "user", "content": [image_content_block(b64)]}],
        )
        self.last_usage = response.usage
        return response_text(response.content)

    def caption(self, path: Path) -> str:
        b64 = encode_image(
            path,
            long_edge=self.settings.caption_resize_long_edge,
            quality=self.settings.caption_jpeg_quality,
        )
        return self._caption_b64(b64)

    def caption_pil(self, img) -> str:
        """Caption an in-memory PIL image (M7 video keyframes)."""
        b64 = encode_pil_image(
            img,
            long_edge=self.settings.caption_resize_long_edge,
            quality=self.settings.caption_jpeg_quality,
        )
        return self._caption_b64(b64)


class CaptionWorker:
    def __init__(
        self,
        db_path: Path,
        media_dir: Path,
        settings: Settings,
        captioner: Captioner | None = None,
        embed_enqueue: Callable[[int], None] | None = None,
    ):
        self.db_path = db_path
        self.media_dir = media_dir
        self.settings = settings
        self.captioner = captioner or Captioner(settings)
        # Hook to enqueue a follow-on embed job once the caption is indexed (M6).
        self.embed_enqueue = embed_enqueue
        self.queue: queue.Queue[int] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="caption", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.queue.put(_SENTINEL)
        if self._thread:
            self._thread.join(timeout=5)

    def enqueue(self, job_id: int) -> None:
        self.queue.put(job_id)

    def _enqueue_embed(self, store: Store, message_id: int) -> None:
        """Queue an embed job for the freshly-captioned message (M6)."""
        if self.embed_enqueue is None:
            return
        job_id = store.create_job(job_type="embed", ref_id=message_id)
        self.embed_enqueue(job_id)

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
        """Re-enqueue work left over from a previous run. Jobs attached to an
        Anthropic batch are NOT stale — the backfill CLI polls those."""
        reset = store.reset_stale_jobs("caption", unbatched_only=True)
        pending = store.pending_jobs("caption")
        if reset or pending:
            log.info("recovered %d stale + %d pending caption jobs", reset, len(pending))
        for job_id in pending:
            self.queue.put(job_id)

    def _process(self, store: Store, job_id: int) -> None:
        ref = store.job_ref(job_id)
        if ref is None:
            return
        _, media_id = ref
        store.claim_job(job_id)
        try:
            media = store.media_row(media_id)
            if media is None or media[1] is None:
                raise FileNotFoundError(f"no media file recorded for media {media_id}")
            message_id, rel_path, _skipped = media
            text = self.captioner.caption(self.media_dir / rel_path)
            usage = getattr(self.captioner, "last_usage", None)
            if usage is not None:
                store.record_spend(model=self.captioner.model, usage=usage)
            if text:
                store.insert_caption(
                    media_id=media_id, text=text, model=self.captioner.model
                )
                store.index_text(message_id, text)
                self._enqueue_embed(store, message_id)
            store.finish_job(job_id, ok=True, max_attempts=self.settings.job_max_attempts)
            log.info("captioned media %s (%d chars)", media_id, len(text))
        except Exception:
            log.exception("caption job %s failed", job_id)
            state = store.finish_job(
                job_id, ok=False, max_attempts=self.settings.job_max_attempts
            )
            if state == "pending":
                self.queue.put(job_id)  # retry
