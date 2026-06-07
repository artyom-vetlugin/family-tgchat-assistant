-- Layer 1: raw, immutable chat archive.
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS senders (
  id           INTEGER PRIMARY KEY,
  tg_user_id   INTEGER UNIQUE,
  display_name TEXT,
  aliases      TEXT  -- JSON array of name variants (from export sender names)
);

CREATE TABLE IF NOT EXISTS messages (
  id            INTEGER PRIMARY KEY,
  tg_message_id INTEGER NOT NULL,
  tg_chat_id    INTEGER NOT NULL,
  sender_id     INTEGER REFERENCES senders(id),
  ts            INTEGER NOT NULL,           -- unix seconds
  reply_to      INTEGER,                    -- parent tg_message_id
  kind          TEXT NOT NULL,              -- text|voice|video_note|photo|video|file|sticker|other
  text          TEXT,                       -- message text or media caption
  source        TEXT NOT NULL DEFAULT 'live',  -- live|export
  UNIQUE(tg_chat_id, tg_message_id, source)
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);
CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender_id);

CREATE TABLE IF NOT EXISTS media (
  id         INTEGER PRIMARY KEY,
  message_id INTEGER NOT NULL REFERENCES messages(id),
  kind       TEXT NOT NULL,                 -- photo|video|voice|video_note|file
  rel_path   TEXT,                          -- under media/ (NULL if not downloaded)
  mime       TEXT,
  bytes      INTEGER,
  skipped    INTEGER NOT NULL DEFAULT 0,    -- 1 if >20MB live download limit
  sha256     TEXT
);
CREATE INDEX IF NOT EXISTS idx_media_message ON media(message_id);

CREATE TABLE IF NOT EXISTS transcripts (
  id         INTEGER PRIMARY KEY,
  message_id INTEGER NOT NULL UNIQUE REFERENCES messages(id),
  text       TEXT NOT NULL,
  engine     TEXT,                          -- e.g. faster-whisper:large-v3-turbo
  lang       TEXT DEFAULT 'ru',
  created_at INTEGER
);

CREATE TABLE IF NOT EXISTS captions (
  id         INTEGER PRIMARY KEY,
  media_id   INTEGER NOT NULL UNIQUE REFERENCES media(id),
  text       TEXT NOT NULL,                 -- Russian description of the image
  model      TEXT,
  created_at INTEGER
);

CREATE TABLE IF NOT EXISTS jobs (
  id         INTEGER PRIMARY KEY,
  job_type   TEXT NOT NULL,                 -- caption|transcribe|video
  ref_id     INTEGER NOT NULL,              -- media.id or messages.id depending on job_type
  state      TEXT NOT NULL DEFAULT 'pending',  -- pending|inflight|done|error
  batch_id   TEXT,                          -- Anthropic Batch API id
  attempts   INTEGER NOT NULL DEFAULT 0,
  updated_at INTEGER,
  UNIQUE(job_type, ref_id)
);

-- Full-text search over message text + transcripts + captions.
-- trigram tokenizer: substring matching, works for Russian without a stemmer.
CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
  body,
  message_id UNINDEXED,
  ts UNINDEXED,
  sender UNINDEXED,
  tokenize = 'trigram'
);
