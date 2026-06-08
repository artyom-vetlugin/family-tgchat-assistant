# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Telegram bot for a Russian-language family group chat: archives all messages
into SQLite, transcribes voice locally, captions photos, and answers questions
via the Claude API (Haiku for bulk work, Sonnet for answers — **no Opus**;
cost minimization is a user-mandated constraint).

**Read `ROADMAP.md` first.** It is the source of truth for scope: full
architecture with rationale for every design decision, milestones M1–M7 with
acceptance criteria, and operational gotchas. Update its status markers when a
milestone lands. Work proceeds milestone by milestone — don't pull future
milestones' scope forward without asking.

## Commands

```bash
uv sync --python 3.12 --extra dev     # install (uses uv, not pip)
uv run python -m family_assistant     # run the bot (needs .env — see .env.example)
uv run python -m family_assistant.backfill [export_dir] [--chat-id ID] [--transcribe]
                                      # one-time history import (M3, idempotent)
uv run python -m family_assistant.caption_backfill [--max-batches N] [--retry-errored]
                                      # caption archived photos via Batch API (M4,
                                      # resumable; --max-batches 1 = cost-capped trial;
                                      # --retry-errored revives dead jobs, corrupt re-skip)
uv run python -m family_assistant.digest [--date YYYY-MM-DD] [--rebuild]
                                      # nightly wiki digest (M5); default = catch up the
                                      # log.md watermark → yesterday; --rebuild re-seeds
                                      # all history month by month (deterministic)
uv run python -m family_assistant.embed_backfill [--limit N] [--retry-errored]
                                      # local semantic-search embeddings (M6, $0); resumable
                                      # inline drain over every message with searchable text;
                                      # --limit N = trial; first run downloads the e5 model
uv run pytest                         # all tests
uv run pytest tests/test_store.py -k trigram   # single test
```

There is no linter configured. Runtime config comes from `.env`
(pydantic-settings, see `src/family_assistant/config.py`).

## Architecture

Flow: aiogram bot (`bot.py`) logs **every** group message to SQLite and, when
@mentioned/replied, hands the question to `query/engine.py`, which makes one
cheap Haiku call (`query/router.py`) to pick an intent — `generic_llm` gets a
direct Sonnet answer; everything else goes to `query/agent.py`, a manual
Sonnet tool-use loop (max 6 iterations, then a forced no-tools final answer)
over retrieval tools backed by `store/db.py`.

Storage follows the LLM-Wiki three-layer pattern: Layer 1 is the immutable
SQLite archive + `media/` files; Layer 2 (`wiki/` markdown, M5) is LLM-maintained
by the nightly digest and fully rebuildable from Layer 1 (`wiki.py` does the
path-validated I/O); Layer 3 (`schema/wiki_guide.md`) is human-edited rules.

The digest (`digest/runner.py`, `python -m family_assistant.digest`) is a sync
Haiku tool-loop that edits wiki pages. Two M5 design choices to preserve:
`index.md` is **auto-generated** from each page's first line (never hand-edited
by the LLM — keeps it deterministic), and the digest **watermark is derived from
`log.md`** (last `## YYYY-MM-DD` heading), not a table — so the wiki stays
self-describing and rebuildable. Cache layout: guide (frozen) → start-of-period
`index.md` snapshot (breakpoint) → period messages in the user turn. `--rebuild`
seeds history with sequential **monthly** passes (not Batch API: each month
builds on the prior month's pages).

`store/schema.sql` already contains the tables for future milestones
(`media`, `transcripts`, `captions`, `jobs`; `embeddings` was added in M6) —
fill them in, don't migrate (schema is re-applied idempotently with
`CREATE … IF NOT EXISTS` on every `Store.__init__`, so a brand-new table just
appears on next start).
Media messages are already logged as rows with empty content; later
milestones attach transcripts/captions via `store.index_text(message_id, text)`,
which makes them searchable (this path is tested).

Jobs (`jobs` table) drive transcription (M2), captioning (M4) and embedding
(M6). Watch the `ref_id` divergence: `transcribe` and `embed` jobs use
`messages.id`, `caption` jobs use `media.id`. Caption jobs submitted to the
Batch API carry `jobs.batch_id`; the live `CaptionWorker` must recover with
`reset_stale_jobs("caption", unbatched_only=True)` so it never steals work
the `caption_backfill` CLI is still polling for.

Semantic search (M6, `embed.py`): a local sentence-transformers worker
(`EmbeddingWorker`, same thread+WAL shape as the others) embeds each message's
`text_for_message` (the COALESCE of text/transcript/caption) into the
`embeddings` table — one float32 BLOB per ~512-token chunk. `embed` jobs are
enqueued at the three `index_text` chokepoints (text in `bot.log_message`; voice
/photo via an `embed_enqueue` hook on the transcribe/caption workers, since their
text lands later). `store.knn` does brute-force numpy cosine kNN ($0, no
sqlite-vec); the `semantic_search` agent tool shares **one** `Embedder` instance
(threaded `BotApp → QueryEngine → RetrievalAgent`, also given to the worker).
Backfill via `embed_backfill` (local inline drain, no Batch API).

## Invariants to preserve

- **Prompt-cache hygiene** (`query/agent.py`): the system prompt and TOOLS
  list must stay byte-stable — never interpolate dates, names, or ids into
  them. Volatile context (current date, asker) goes into the user message.
  The cache breakpoint sits on the system block and covers tools+system.
- **FTS5 uses the `trigram` tokenizer** (substring matching — the deliberate
  choice for Russian, which the default tokenizer can't stem). Queries under
  3 chars return `[]` instead of raising; user text is quote-escaped before
  MATCH. The agent prompt tells Sonnet to retry with different word forms.
- **Dedup key** is `UNIQUE(tg_chat_id, tg_message_id, source)` with
  `source ∈ {live, export}` — the M3 backfill depends on this.
- DB access is synchronous sqlite3 from the event loop — intentional
  (family-chat traffic is tiny); don't introduce an async DB layer.
- Bot answers in Russian; tool results and stored text are Russian.

## Telegram specifics

- The bot must have **privacy mode disabled** (BotFather) or it silently
  stops receiving regular group messages — `/stats` count growing is the
  health check.
- Bot API can't read pre-join history (backfill comes from Telegram Desktop
  export, M3) and caps file downloads at 20MB (voice/circles always fit;
  bigger videos get `media.skipped=1`).
- Runs on the user's Mac via launchd (`deploy/`); Telegram holds ~24h of
  undelivered updates across sleep, so gaps self-heal on wake. Backlog
  @mentions older than `ANSWER_MAX_AGE_MINUTES` (default 30) are archived but
  not answered; on startup the bot warns if the archive gap exceeds 24h
  (recover via an export backfill re-run).
