# Family Telegram Chat Assistant

A Telegram bot that lives in the family group chat, archives everything —
texts, voice/video-circle and video transcripts, photo captions, plus a
one-time backfill of the chat's full history — and answers questions about it
using the Claude API.

## Current status — M1–M7 complete

- ✅ Logs all group text messages into SQLite (+ FTS5 trigram search, works for Russian)
- ✅ Answers when @mentioned or replied to: Haiku intent router → Sonnet agentic
  retrieval loop (`fts_search`, `get_messages_around`, `recent_window`,
  `read_wiki_page`, `list_wiki_index`, `semantic_search`)
- ✅ Generic LLM questions answered directly
- ✅ M2: voice messages & video circles transcribed locally (faster-whisper,
  $0) and searchable; resumable job queue
- ✅ M3: full history imported from a Telegram Desktop export (dedup against
  live messages, media copied, voice backlog transcribed locally)
- ✅ M4: photos captioned (Haiku vision; live worker + resumable Batch-API backfill)
- ✅ M5: LLM wiki (`wiki/`) maintained by a nightly Haiku digest; the answer
  agent reads it (`list_wiki_index` / `read_wiki_page`)
- ✅ M6: local semantic search (sentence-transformers e5 embeddings, $0) — finds
  paraphrases FTS misses (`semantic_search` tool)
- ✅ M7: videos transcribed + keyframe-captioned and searchable; chat commands
  (`/find`, `/wiki`, `/summary`, `/spend`); router shortcut for obvious queries;
  per-call token-spend tracking; opt-in weekly chat recap

> **Full scope, architecture, design decisions and per-milestone acceptance
> criteria live in [ROADMAP.md](ROADMAP.md)** — read that first when resuming
> work on this project.

## Setup

1. **Create the bot**: talk to [@BotFather](https://t.me/BotFather) → `/newbot`.
   Then **disable privacy mode**: `/mybots` → your bot → Bot Settings →
   Group Privacy → Turn off. (Otherwise the bot won't see regular group messages.)

2. **Configure**:

   ```bash
   cp .env.example .env
   # fill in TELEGRAM_BOT_TOKEN and ANTHROPIC_API_KEY
   ```

3. **Install & run** (needs [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync --python 3.12
   uv run python -m family_assistant
   ```

4. **Add the bot to the family chat.** Send `/id` in the chat, put the printed
   chat id into `.env` as `ALLOWED_CHAT_IDS=[-100123456789]`, restart the bot.

5. Talk to it: `@your_bot что мы обсуждали вчера?`

## Commands

Anyone in the chat can use these (no @mention needed):

- `/find <query>` — full-text search over the archive
- `/wiki [topic]` — read a wiki page, or the index if no topic is given
- `/summary [period]` — recap stitched from the wiki journal (falls back to the agent)
- `/spend` — this month's estimated Anthropic token cost
- `/id`, `/stats` — chat id and archive health (message count, newest message time)

## Backfill chat history (one-time)

The Bot API can't read messages sent before the bot joined. Export the chat
from **Telegram Desktop** (open the chat → ⋮ → Export chat history → format
**JSON**, include media), put the result under `export/` (so that
`export/result.json` exists), then:

```bash
uv run python -m family_assistant.backfill              # import + queue transcription
uv run python -m family_assistant.backfill --transcribe  # …or transcribe inline (hours)
```

The import is idempotent (re-running it is safe), prefers live rows over
export duplicates, and prints a reconciliation report. Without `--transcribe`,
the queued voice messages are transcribed by the bot the next time it runs.

Partial exports work too: text is always present in the JSON regardless of
which media types you tick, and you can import in stages — re-running with a
fuller export (wider date range, more media types) adds the older messages
and fills in previously missing media files in place.

After importing, three more one-time jobs enrich the backfilled media. All are
resumable (re-running skips finished work) and process the archive in place:

```bash
uv run python -m family_assistant.caption_backfill   # photos → Haiku captions (Batch API, 50% off)
uv run python -m family_assistant.embed_backfill      # all text → semantic-search embeddings (local, $0)
uv run python -m family_assistant.video_backfill      # videos → transcript + keyframe captions ($0 whisper)
```

Each takes `--limit N` / `--max-batches N` for a cost-capped trial run; see
[ROADMAP.md](ROADMAP.md) and `CLAUDE.md` for the full flags.

## Wiki digest (M5)

The nightly digest reads each day's messages (texts, transcripts, captions) and
maintains a small LLM-kept wiki under `wiki/` — `people/`, `topics/`, an
append-only `log.md` journal, and an auto-generated `index.md`. The answer agent
consults it first for "что обсуждали на прошлой неделе" / "что нового у X". The
wiki is fully rebuildable from the SQLite archive, so `wiki/` is gitignored;
only the human-edited rules (`schema/wiki_guide.md`, Layer 3) are in git.

```bash
uv run python -m family_assistant.digest             # catch up to yesterday (daily)
uv run python -m family_assistant.digest --rebuild   # seed all history (month by month)
uv run python -m family_assistant.digest --date 2024-05-10   # (re)digest one day
```

Run it without `--rebuild` first only after seeding history once with
`--rebuild` (a fresh wiki otherwise digests just yesterday). It uses Haiku and
prompt caching; the cost is roughly one Haiku pass per day.

## Run as a service (launchd, auto-start + keepalive)

```bash
sed "s|__PROJECT_DIR__|$(pwd)|g; s|__UV__|$(which uv)|g" deploy/com.family.tgassistant.plist \
  > ~/Library/LaunchAgents/com.family.tgassistant.plist
launchctl load ~/Library/LaunchAgents/com.family.tgassistant.plist
# logs:
tail -f data/bot.log
```

The nightly digest runs as a second launchd job (one-shot at 03:00, not
KeepAlive):

```bash
sed "s|__PROJECT_DIR__|$(pwd)|g; s|__UV__|$(which uv)|g" deploy/com.family.tgdigest.plist \
  > ~/Library/LaunchAgents/com.family.tgdigest.plist
launchctl load ~/Library/LaunchAgents/com.family.tgdigest.plist
tail -f data/digest.log
```

If the Mac is asleep at 03:00, launchd runs the digest on the next wake; missed
nights self-heal because the digest catches up from the `log.md` watermark to
yesterday. To run reliably overnight, schedule a wake with e.g.
`sudo pmset repeat wakeorpoweron MTWRFSU 02:55:00`.

The optional weekly chat recap (M7) is a third one-shot job (Sunday 18:00). It's
off by default — set `WEEKLY_SUMMARY_ENABLED=true` in `.env` to enable, then:

```bash
sed "s|__PROJECT_DIR__|$(pwd)|g; s|__UV__|$(which uv)|g" deploy/com.family.tgweekly.plist \
  > ~/Library/LaunchAgents/com.family.tgweekly.plist
launchctl load ~/Library/LaunchAgents/com.family.tgweekly.plist
tail -f data/weekly.log
```

It posts a 7-day recap stitched from `wiki/log.md` (so it needs the digest to
have been running). Same wake caveat as the digest — a missed Sunday posts late.

Note: when the Mac sleeps, Telegram holds undelivered updates for ~24h and the
bot catches up on wake: the backlog is archived (and voice transcribed), but
questions older than 30 minutes (`ANSWER_MAX_AGE_MINUTES`) are not answered —
no ghost replies hours later. If a gap exceeds 24h the startup log warns about
it (those messages are recoverable by re-running the backfill with a fresh
export). If the Mac regularly sleeps longer than a day, consider `caffeinate`
or a `pmset` wake schedule.

## Tests

```bash
uv run pytest
```

## Architecture (short)

```
                          ┌─ Layer 1: SQLite (WAL) + FTS5 trigram + media/  (raw archive)
aiogram bot ──► ingest ───┼─ voice/video → whisper · photos → Haiku caption · video → both
     │                    ├─ Layer 2: wiki/ markdown  (nightly Haiku digest, rebuildable)
     │                    └─ embeddings (local e5)  → semantic search
     │
     └─ @mention ──► Haiku router (or shortcut) ──► Sonnet tool-use loop ──► answer (RU)
                        (intent)      fts_search / semantic_search / wiki / context / recent
```

Cost design: Haiku for routing/bulk (captions, digest), Sonnet only for
user-facing answers, no Opus; prompt caching on the agent's frozen system+tools
prefix; local processing (whisper, e5 embeddings) keeps the heavy media work at
$0. Token spend is tracked per call and reported by `/spend`.
