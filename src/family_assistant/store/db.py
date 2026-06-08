"""SQLite store: raw archive (Layer 1) + FTS5 search index.

Traffic in a 4-person family chat is tiny, so synchronous sqlite3 calls from
the event loop are fine (sub-millisecond inserts in WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass
class MessageRow:
    id: int
    tg_message_id: int
    tg_chat_id: int
    sender: str | None
    ts: int
    reply_to: int | None
    kind: str
    text: str | None

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.ts, tz=timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M"
        )

    def format(self) -> str:
        if self.text:
            if self.kind in ("voice", "video_note"):
                prefix = "[голосовое] "
            elif self.kind == "photo":
                prefix = "[фото] "
            elif self.kind == "video":
                prefix = "[видео] "
            else:
                prefix = ""
            body = prefix + self.text
        else:
            body = f"[{self.kind} без текста]"
        return f"[{self.when}] {self.sender or '?'} (msg {self.tg_message_id}): {body}"


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # --- senders -----------------------------------------------------------

    def upsert_sender(self, tg_user_id: int, display_name: str) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO senders (tg_user_id, display_name) VALUES (?, ?)
            ON CONFLICT(tg_user_id) DO UPDATE SET display_name = excluded.display_name
            RETURNING id
            """,
            (tg_user_id, display_name),
        )
        sender_id = cur.fetchone()[0]
        self.conn.commit()
        return sender_id

    def upsert_export_sender(
        self, tg_user_id: int | None, name: str | None
    ) -> int | None:
        """Resolve an export sender without clobbering the live display_name.

        The bot's `upsert_sender` overwrites display_name (live names are
        authoritative); export name variants are accumulated in `aliases`.
        """
        if tg_user_id is None:
            return None
        row = self.conn.execute(
            "SELECT id, display_name, aliases FROM senders WHERE tg_user_id = ?",
            (tg_user_id,),
        ).fetchone()
        if row is None:
            cur = self.conn.execute(
                "INSERT INTO senders (tg_user_id, display_name, aliases) VALUES (?, ?, ?)",
                (tg_user_id, name, json.dumps([])),
            )
            self.conn.commit()
            return cur.lastrowid
        if name and name != row["display_name"]:
            aliases = parse_aliases(row["aliases"])
            if name not in aliases:
                aliases.append(name)
                self.conn.execute(
                    "UPDATE senders SET aliases = ? WHERE id = ?",
                    (json.dumps(aliases, ensure_ascii=False), row["id"]),
                )
                self.conn.commit()
        return row["id"]

    # --- messages ----------------------------------------------------------

    def insert_message(
        self,
        *,
        tg_message_id: int,
        tg_chat_id: int,
        sender_id: int | None,
        ts: int,
        kind: str,
        text: str | None,
        reply_to: int | None = None,
        source: str = "live",
    ) -> int | None:
        """Insert a message; returns row id, or None if it was a duplicate."""
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO messages
              (tg_message_id, tg_chat_id, sender_id, ts, reply_to, kind, text, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tg_message_id, tg_chat_id, sender_id, ts, reply_to, kind, text, source),
        )
        self.conn.commit()
        return cur.lastrowid if cur.rowcount else None

    def live_message_id(self, tg_chat_id: int, tg_message_id: int) -> int | None:
        """Row id of the live-source row for this telegram message, if any."""
        row = self.conn.execute(
            "SELECT id FROM messages "
            "WHERE tg_chat_id = ? AND tg_message_id = ? AND source = 'live'",
            (tg_chat_id, tg_message_id),
        ).fetchone()
        return row["id"] if row else None

    def export_message_id(self, tg_chat_id: int, tg_message_id: int) -> int | None:
        """Row id of the export-source row for this telegram message, if any."""
        row = self.conn.execute(
            "SELECT id FROM messages "
            "WHERE tg_chat_id = ? AND tg_message_id = ? AND source = 'export'",
            (tg_chat_id, tg_message_id),
        ).fetchone()
        return row["id"] if row else None

    def live_messages_at(self, tg_chat_id: int, ts: int) -> list[tuple[int, str | None]]:
        """(id, text) of live rows at an exact timestamp — for fallback dedup."""
        rows = self.conn.execute(
            "SELECT id, text FROM messages "
            "WHERE tg_chat_id = ? AND ts = ? AND source = 'live'",
            (tg_chat_id, ts),
        ).fetchall()
        return [(r["id"], r["text"]) for r in rows]

    def text_for_message(self, message_id: int) -> str | None:
        """The searchable text for one message: message text, else transcript,
        else caption (same COALESCE source as the retrieval tools) — what the
        embedding worker turns into vectors."""
        row = self.conn.execute(
            """
            SELECT COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
            FROM messages m
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE m.id = ?
            """,
            (message_id,),
        ).fetchone()
        return row["text"] if row else None

    def index_text(self, message_id: int, body: str) -> None:
        """Add searchable text (message text, transcript, or caption) to FTS."""
        row = self.conn.execute(
            """
            SELECT m.ts, s.display_name AS sender
            FROM messages m LEFT JOIN senders s ON s.id = m.sender_id
            WHERE m.id = ?
            """,
            (message_id,),
        ).fetchone()
        if row is None:
            return
        self.conn.execute(
            "INSERT INTO search (body, message_id, ts, sender) VALUES (?, ?, ?, ?)",
            (body, message_id, row["ts"], row["sender"]),
        )
        self.conn.commit()

    # --- media / transcripts / jobs (M2+) ------------------------------------

    def insert_media(
        self,
        *,
        message_id: int,
        kind: str,
        rel_path: str | None,
        mime: str | None = None,
        bytes: int | None = None,
        skipped: bool = False,
        sha256: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO media (message_id, kind, rel_path, mime, bytes, skipped, sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, kind, rel_path, mime, bytes, int(skipped), sha256),
        )
        self.conn.commit()
        return cur.lastrowid

    def has_media(self, message_id: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM media WHERE message_id = ? LIMIT 1", (message_id,)
        ).fetchone()
        return row is not None

    def upgrade_media(
        self,
        message_id: int,
        *,
        rel_path: str,
        mime: str | None = None,
        bytes: int | None = None,
        sha256: str | None = None,
    ) -> None:
        """Replace a placeholder media row (rel_path NULL / skipped) with the
        real file — e.g. a fuller export now includes a previously missing file."""
        self.conn.execute(
            "UPDATE media SET rel_path = ?, mime = ?, bytes = ?, skipped = 0, sha256 = ? "
            "WHERE message_id = ?",
            (rel_path, mime, bytes, sha256, message_id),
        )
        self.conn.commit()

    def media_for_message(self, message_id: int) -> tuple[str | None, bool] | None:
        """Returns (rel_path, skipped) for the message's media, or None."""
        row = self.conn.execute(
            "SELECT rel_path, skipped FROM media WHERE message_id = ? LIMIT 1",
            (message_id,),
        ).fetchone()
        return (row["rel_path"], bool(row["skipped"])) if row else None

    def insert_transcript(
        self, *, message_id: int, text: str, engine: str, lang: str = "ru"
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO transcripts (message_id, text, engine, lang, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, text, engine, lang, int(time.time())),
        )
        self.conn.commit()

    def media_row(self, media_id: int) -> tuple[int, str | None, bool] | None:
        """Returns (message_id, rel_path, skipped) for a media row, or None."""
        row = self.conn.execute(
            "SELECT message_id, rel_path, skipped FROM media WHERE id = ?",
            (media_id,),
        ).fetchone()
        return (row["message_id"], row["rel_path"], bool(row["skipped"])) if row else None

    def insert_caption(self, *, media_id: int, text: str, model: str) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO captions (media_id, text, model, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (media_id, text, model, int(time.time())),
        )
        self.conn.commit()

    def photo_media_without_caption(self) -> list[int]:
        """Media ids of downloaded photos that have no caption yet (M4 backfill)."""
        rows = self.conn.execute(
            """
            SELECT m.id FROM media m
            LEFT JOIN captions c ON c.media_id = m.id
            WHERE m.kind = 'photo' AND m.rel_path IS NOT NULL AND m.skipped = 0
              AND c.id IS NULL
            ORDER BY m.id
            """
        ).fetchall()
        return [r["id"] for r in rows]

    def video_media_without_transcript(self) -> list[int]:
        """message_ids of downloaded videos with no transcript yet (M7 backfill).

        Video jobs use ref_id = messages.id, so this returns message ids. Export
        videos land as media rows with no job; this is the backfill enqueue source
        (resumable: a transcript already present makes a re-run skip the row)."""
        rows = self.conn.execute(
            """
            SELECT m.message_id FROM media m
            LEFT JOIN transcripts tr ON tr.message_id = m.message_id
            WHERE m.kind = 'video' AND m.rel_path IS NOT NULL AND m.skipped = 0
              AND tr.id IS NULL
            ORDER BY m.message_id
            """
        ).fetchall()
        return [r["message_id"] for r in rows]

    def create_job(self, *, job_type: str, ref_id: int) -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO jobs (job_type, ref_id, updated_at) VALUES (?, ?, ?)",
            (job_type, ref_id, int(time.time())),
        )
        self.conn.commit()
        if cur.rowcount:
            return cur.lastrowid
        # Job already existed (UNIQUE(job_type, ref_id)) — return the existing id.
        row = self.conn.execute(
            "SELECT id FROM jobs WHERE job_type = ? AND ref_id = ?",
            (job_type, ref_id),
        ).fetchone()
        return row["id"]

    def job_ref(self, job_id: int) -> tuple[str, int] | None:
        row = self.conn.execute(
            "SELECT job_type, ref_id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return (row["job_type"], row["ref_id"]) if row else None

    def claim_job(self, job_id: int) -> int:
        """Mark the job inflight; returns the new attempts count."""
        cur = self.conn.execute(
            """
            UPDATE jobs SET state = 'inflight', attempts = attempts + 1, updated_at = ?
            WHERE id = ? RETURNING attempts
            """,
            (int(time.time()), job_id),
        )
        attempts = cur.fetchone()[0]
        self.conn.commit()
        return attempts

    def finish_job(self, job_id: int, *, ok: bool, max_attempts: int) -> str:
        """Finalize a job attempt; returns the resulting state
        ('done', 'pending' for retry, or 'error' when attempts exhausted)."""
        if ok:
            state = "done"
        else:
            row = self.conn.execute(
                "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            state = "error" if row and row["attempts"] >= max_attempts else "pending"
        self.conn.execute(
            "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?",
            (state, int(time.time()), job_id),
        )
        self.conn.commit()
        return state

    def pending_jobs(self, job_type: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM jobs WHERE job_type = ? AND state = 'pending' ORDER BY id",
            (job_type,),
        ).fetchall()
        return [r["id"] for r in rows]

    def set_job_batch(self, job_id: int, batch_id: str) -> None:
        """Record the Anthropic Batch API id so a killed run can resume polling."""
        self.conn.execute(
            "UPDATE jobs SET batch_id = ?, updated_at = ? WHERE id = ?",
            (batch_id, int(time.time()), job_id),
        )
        self.conn.commit()

    def batch_ids_for_jobs(self, job_type: str) -> list[str]:
        """Batch ids still attached to inflight jobs (in-progress Batch API work)."""
        rows = self.conn.execute(
            "SELECT DISTINCT batch_id FROM jobs "
            "WHERE job_type = ? AND state = 'inflight' AND batch_id IS NOT NULL "
            "ORDER BY batch_id",
            (job_type,),
        ).fetchall()
        return [r["batch_id"] for r in rows]

    def job_state(self, job_id: int) -> tuple[str, str | None] | None:
        """(state, batch_id) for a job, or None."""
        row = self.conn.execute(
            "SELECT state, batch_id FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return (row["state"], row["batch_id"]) if row else None

    def reset_stale_jobs(self, job_type: str, *, unbatched_only: bool = False) -> int:
        """Recover jobs left 'inflight' by a previous run (crash/restart).

        unbatched_only=True leaves jobs attached to an Anthropic batch alone —
        those are legitimately in flight at the API, not stale (the caption
        worker must not steal work the backfill CLI is still polling for).
        """
        sql = (
            "UPDATE jobs SET state = 'pending', updated_at = ? "
            "WHERE job_type = ? AND state = 'inflight'"
        )
        if unbatched_only:
            sql += " AND batch_id IS NULL"
        cur = self.conn.execute(sql, (int(time.time()), job_type))
        self.conn.commit()
        return cur.rowcount

    def reset_errored_jobs(self, job_type: str) -> int:
        """Reset terminally-errored jobs back to 'pending' so a re-run retries them.

        attempts is zeroed so a transient failure gets the full retry budget
        again; a corrupt image simply re-skips on its next attempt (one attempt
        via the encode-skip path), so this never loops.
        """
        cur = self.conn.execute(
            "UPDATE jobs SET state = 'pending', attempts = 0, updated_at = ? "
            "WHERE job_type = ? AND state = 'error'",
            (int(time.time()), job_type),
        )
        self.conn.commit()
        return cur.rowcount

    # --- embeddings (M6 semantic search) ------------------------------------

    def replace_embeddings(
        self, message_id: int, vectors: np.ndarray, *, model: str, dim: int
    ) -> None:
        """Store one vector per chunk for a message, replacing any existing ones
        for this model (re-embedding is idempotent). `vectors` is (n_chunks, dim),
        L2-normalized float32."""
        self.conn.execute(
            "DELETE FROM embeddings WHERE message_id = ? AND model = ?",
            (message_id, model),
        )
        now = int(time.time())
        rows = [
            (message_id, i, vec.astype(np.float32).tobytes(), model, dim, now)
            for i, vec in enumerate(vectors)
        ]
        self.conn.executemany(
            """
            INSERT INTO embeddings (message_id, chunk_index, vector, model, dim, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def messages_needing_embedding(self, model: str) -> list[int]:
        """Message ids that have searchable text but no embedding for `model` —
        the backfill enqueue source (resumable: re-runs skip embedded rows)."""
        rows = self.conn.execute(
            """
            SELECT m.id FROM messages m
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE COALESCE(m.text, tr.text,
                    (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                     WHERE md.message_id = m.id LIMIT 1)) IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM embeddings e
                WHERE e.message_id = m.id AND e.model = ?
              )
            ORDER BY m.id
            """,
            (model,),
        ).fetchall()
        return [r["id"] for r in rows]

    def knn(
        self,
        query_vec: np.ndarray,
        *,
        model: str,
        k: int = 10,
        sender: str | None = None,
        after_ts: int | None = None,
        before_ts: int | None = None,
    ) -> list[MessageRow]:
        """Brute-force cosine kNN over stored vectors (query_vec L2-normalized).

        Loads the candidate vectors (optionally filtered by sender/time), scores
        them with a single matmul, dedups chunks back to their message keeping the
        best score, and returns the top-k as MessageRows — same shape as `search`."""
        sql = """
            SELECT e.message_id, e.vector
            FROM embeddings e
            JOIN messages m ON m.id = e.message_id
            LEFT JOIN senders se ON se.id = m.sender_id
            WHERE e.model = ?
        """
        params: list = [model]
        if sender:
            sql += " AND se.display_name LIKE ?"
            params.append(f"%{sender}%")
        if after_ts:
            sql += " AND m.ts >= ?"
            params.append(after_ts)
        if before_ts:
            sql += " AND m.ts <= ?"
            params.append(before_ts)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return []

        matrix = np.frombuffer(
            b"".join(r["vector"] for r in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        scores = matrix @ query_vec.astype(np.float32)

        best: dict[int, float] = {}
        for r, score in zip(rows, scores):
            mid = r["message_id"]
            if score > best.get(mid, -np.inf):
                best[mid] = float(score)
        top_ids = sorted(best, key=lambda mid: best[mid], reverse=True)[:k]
        if not top_ids:
            return []

        placeholders = ",".join("?" for _ in top_ids)
        fetched = self.conn.execute(
            f"""
            SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                   m.ts, m.reply_to, m.kind, COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
            FROM messages m LEFT JOIN senders se ON se.id = m.sender_id
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE m.id IN ({placeholders})
            """,
            top_ids,
        ).fetchall()
        by_id = {r["id"]: self._to_message_row(r) for r in fetched}
        return [by_id[mid] for mid in top_ids if mid in by_id]

    # --- retrieval (query-engine tools) -------------------------------------

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        sender: str | None = None,
        after_ts: int | None = None,
        before_ts: int | None = None,
    ) -> list[MessageRow]:
        # Quote the query so user text can't break FTS5 syntax; trigram tokenizer
        # then does substring matching on the quoted string.
        fts_query = '"' + query.replace('"', '""') + '"'
        sql = """
            SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                   m.ts, m.reply_to, m.kind,
                   COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
            FROM search s
            JOIN messages m ON m.id = s.message_id
            LEFT JOIN senders se ON se.id = m.sender_id
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE search MATCH ?
        """
        params: list = [fts_query]
        if sender:
            sql += " AND se.display_name LIKE ?"
            params.append(f"%{sender}%")
        if after_ts:
            sql += " AND m.ts >= ?"
            params.append(after_ts)
        if before_ts:
            sql += " AND m.ts <= ?"
            params.append(before_ts)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            rows = self.conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []  # malformed query (e.g. <3 chars for trigram)
        return [self._to_message_row(r) for r in rows]

    def get_messages_around(
        self, tg_chat_id: int, tg_message_id: int, n: int = 5
    ) -> list[MessageRow]:
        anchor = self.conn.execute(
            "SELECT ts FROM messages WHERE tg_chat_id = ? AND tg_message_id = ? LIMIT 1",
            (tg_chat_id, tg_message_id),
        ).fetchone()
        if anchor is None:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM (
              SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                     m.ts, m.reply_to, m.kind, COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
              FROM messages m LEFT JOIN senders se ON se.id = m.sender_id
              LEFT JOIN transcripts tr ON tr.message_id = m.id
              WHERE m.tg_chat_id = ? AND m.ts <= ? ORDER BY m.ts DESC LIMIT ?
            )
            UNION
            SELECT * FROM (
              SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                     m.ts, m.reply_to, m.kind, COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
              FROM messages m LEFT JOIN senders se ON se.id = m.sender_id
              LEFT JOIN transcripts tr ON tr.message_id = m.id
              WHERE m.tg_chat_id = ? AND m.ts > ? ORDER BY m.ts ASC LIMIT ?
            )
            ORDER BY ts
            """,
            (tg_chat_id, anchor["ts"], n + 1, tg_chat_id, anchor["ts"], n),
        ).fetchall()
        return [self._to_message_row(r) for r in rows]

    def recent_window(self, tg_chat_id: int, hours: int, limit: int = 200) -> list[MessageRow]:
        since = int(time.time()) - hours * 3600
        rows = self.conn.execute(
            """
            SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                   m.ts, m.reply_to, m.kind, COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
            FROM messages m LEFT JOIN senders se ON se.id = m.sender_id
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE m.tg_chat_id = ? AND m.ts >= ?
            ORDER BY m.ts DESC LIMIT ?
            """,
            (tg_chat_id, since, limit),
        ).fetchall()
        return [self._to_message_row(r) for r in reversed(rows)]

    def messages_between(
        self, tg_chat_id: int, start_ts: int, end_ts: int, limit: int = 5000
    ) -> list[MessageRow]:
        """Messages in [start_ts, end_ts) chronologically, with resolved text
        (message text, else transcript, else caption) — the digest's day window."""
        rows = self.conn.execute(
            """
            SELECT m.id, m.tg_message_id, m.tg_chat_id, se.display_name AS sender,
                   m.ts, m.reply_to, m.kind, COALESCE(m.text, tr.text,
                     (SELECT c.text FROM captions c JOIN media md ON md.id = c.media_id
                      WHERE md.message_id = m.id LIMIT 1)) AS text
            FROM messages m LEFT JOIN senders se ON se.id = m.sender_id
            LEFT JOIN transcripts tr ON tr.message_id = m.id
            WHERE m.tg_chat_id = ? AND m.ts >= ? AND m.ts < ?
            ORDER BY m.ts ASC LIMIT ?
            """,
            (tg_chat_id, start_ts, end_ts, limit),
        ).fetchall()
        return [self._to_message_row(r) for r in rows]

    def first_message_date(self, tg_chat_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT MIN(ts) AS first_ts FROM messages WHERE tg_chat_id = ?",
            (tg_chat_id,),
        ).fetchone()
        return row["first_ts"] if row and row["first_ts"] is not None else None

    def stats(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts FROM messages"
        ).fetchone()
        return dict(row)

    # --- token spend accounting (M7) ----------------------------------------

    def record_spend(self, *, model: str, usage, day: str | None = None) -> None:
        """Accumulate one Anthropic call's token usage into the `spend` table.

        `usage` is any object exposing the Anthropic usage fields; cache fields
        are read defensively (the router's json_schema call and generic answers
        may omit them). `day` defaults to the local date. Called at every API
        call site so `/spend` can report the month."""
        if day is None:
            day = datetime.now().astimezone().strftime("%Y-%m-%d")
        in_tokens = getattr(usage, "input_tokens", 0) or 0
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        out_tokens = getattr(usage, "output_tokens", 0) or 0
        self.conn.execute(
            """
            INSERT INTO spend
              (day, model, in_tokens, cached_tokens, cache_write_tokens, out_tokens, calls)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(day, model) DO UPDATE SET
              in_tokens          = in_tokens + excluded.in_tokens,
              cached_tokens      = cached_tokens + excluded.cached_tokens,
              cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
              out_tokens         = out_tokens + excluded.out_tokens,
              calls              = calls + 1
            """,
            (day, model, in_tokens, cached, cache_write, out_tokens),
        )
        self.conn.commit()

    def spend_summary(self, month: str) -> list[dict]:
        """Per-model token totals for a 'YYYY-MM' month, for the `/spend` report."""
        rows = self.conn.execute(
            """
            SELECT model,
                   SUM(in_tokens)          AS in_tokens,
                   SUM(cached_tokens)      AS cached_tokens,
                   SUM(cache_write_tokens) AS cache_write_tokens,
                   SUM(out_tokens)         AS out_tokens,
                   SUM(calls)              AS calls
            FROM spend
            WHERE day LIKE ?
            GROUP BY model
            ORDER BY model
            """,
            (month + "-%",),
        ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def _to_message_row(r: sqlite3.Row) -> MessageRow:
        return MessageRow(
            id=r["id"],
            tg_message_id=r["tg_message_id"],
            tg_chat_id=r["tg_chat_id"],
            sender=r["sender"],
            ts=r["ts"],
            reply_to=r["reply_to"],
            kind=r["kind"],
            text=r["text"],
        )


def format_rows(rows: list[MessageRow]) -> str:
    if not rows:
        return "(ничего не найдено)"
    return "\n".join(r.format() for r in rows)


def parse_aliases(raw: str | None) -> list[str]:
    return json.loads(raw) if raw else []
