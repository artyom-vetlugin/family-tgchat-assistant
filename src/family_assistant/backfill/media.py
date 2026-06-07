"""Copy export media files into the project's media/ tree.

Layout mirrors the live one (`live/YYYY/MM/<msgid>.<ext>` in bot.py):
exported files land in `export/YYYY/MM/<msgid><orig-ext>`, so the archive is
self-contained and the export dir can be deleted after a successful run.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class CopiedMedia:
    rel_path: str  # under media_dir, e.g. "export/2023/05/100.ogg"
    sha256: str
    bytes: int
    reused: bool  # destination already existed with identical content


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_media(
    *, export_dir: Path, media_dir: Path, src_rel: str, msg_id: int, ts: int
) -> CopiedMedia | None:
    """Copy one export file into media/export/; returns None if it's missing.

    Re-run safe: if the destination already exists with the same sha256, the
    copy is skipped and the existing file is reused.
    """
    src = export_dir / src_rel
    if not src.is_file():
        log.warning("export media missing on disk: %s", src)
        return None
    when = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    rel_path = f"export/{when:%Y/%m}/{msg_id}{Path(src_rel).suffix}"
    dest = media_dir / rel_path
    src_hash = sha256_file(src)
    if dest.is_file() and sha256_file(dest) == src_hash:
        return CopiedMedia(rel_path=rel_path, sha256=src_hash, bytes=dest.stat().st_size, reused=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return CopiedMedia(rel_path=rel_path, sha256=src_hash, bytes=dest.stat().st_size, reused=False)
