# Roadmap & Design Document

This is the **source of truth** for the project scope. If you (human or AI
assistant) are picking this project up fresh, read this file first — it
contains the full architecture, all design decisions with their rationale,
and the per-milestone scope with acceptance criteria.

## Problem & Goal

A family Telegram group (4 members, Russian-language) has accumulated years
of content: ~3,041 voice messages, ~2,174 photos, 227 videos, video circles
(video notes), 508 files, 258 links, plus heavy text traffic. Keeping that
context in human memory is no longer feasible.

**Goal:** a bot in the chat that continuously archives everything
(transcribing voice/video circles, captioning images), organizes knowledge
using Karpathy's **LLM Wiki** pattern
([gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)),
and answers family members' questions: search, summaries, context Q&A, and
generic LLM questions — via the Claude API.

**Cost minimization is a first-class constraint** (target: ~$5–15 one-time,
~$3–10/month ongoing).

## Confirmed constraints & decisions

| Decision | Choice | Why |
|---|---|---|
| Language of chat | Russian only | faster-whisper handles Russian excellently → local transcription is free |
| History backfill | Telegram Desktop JSON export | Bot API **cannot** read pre-join history; export bypasses the 20MB download limit |
| Hosting | User's own Mac (Apple Silicon) | Free; local whisper + local embeddings; bot offline during sleep is acceptable (Telegram holds updates ~24h) |
| Stack | Python 3.12, uv, aiogram 3.x, anthropic SDK, SQLite | |
| Models | `claude-haiku-4-5` for bulk (captions, digest, routing), `claude-sonnet-4-6` for user-facing answers | Cost; **no Opus** (user-approved) |
| FTS tokenizer | FTS5 `trigram` | Default tokenizer does no Russian stemming ("поехали" ≠ "поехать"); trigram = substring match, zero deps. Gaps closed by agentic re-querying + embeddings (M6) |
| Wiki maintenance | Nightly batch digest, NOT per-message | ~30 Haiku calls/month instead of thousands (~100× cheaper); live messages stay searchable via FTS immediately |
| Retrieval | Agentic tool-use loop, NOT context stuffing | Full history exceeds any context window; tools pull only relevant KBs → cents per query |
| Telegram | Privacy mode disabled (or bot is admin) | Required to receive all group messages |

## Architecture

```
Telegram Bot (aiogram 3.x, long polling)
  └─ receives group messages + media; answers @mentions / replies
        │
Ingestion pipeline
  text         → store directly                          [M1 ✅]
  voice/circle → faster-whisper large-v3-turbo (local)   [M2 ✅]
  image        → Haiku vision caption (Batch API)        [M4 ✅]
  text/voice/image → e5 local embeddings (semantic search) [M6 ✅]
  video        → ffmpeg audio → whisper (+ keyframes)    [M7]
        │
Storage — three LLM-Wiki layers
  Layer 1: SQLite (WAL) + FTS5 trigram + media/   ← raw, immutable [M1 ✅]
  Layer 2: wiki/ markdown (index.md, log.md, people/, topics/)
           ← LLM-maintained, fully rebuildable from Layer 1        [M5 ✅]
  Layer 3: schema/wiki_guide.md ← human-edited maintenance rules   [M5 ✅]
        │
Query engine [M1 ✅]
  Haiku intent router → search_history | summarize | context_qa → Sonnet agentic loop
                      → generic_llm → direct Sonnet answer
  Agent tools: fts_search, get_messages_around, recent_window      [M1 ✅]
               read_wiki_page, list_wiki_index                     [M5 ✅]
               semantic_search                                     [M6 ✅]
        │
Nightly digest job (Haiku, launchd timer) → maintains wiki + log.md [M5 ✅]
Backfill job (one-time) → parses Desktop export result.json        [M3]
```

### Cost tactics (apply everywhere)

1. **Prompt caching**: frozen system prompt + tools (+ later `wiki_guide.md`,
   `index.md`) before the cache breakpoint; volatile content (question, day's
   messages) after. Verify via `usage.cache_read_input_tokens`. Never
   interpolate timestamps/names into the system prompt.
2. **Haiku for bulk**, Sonnet only for user-facing answers, no Opus.
3. **Batch API (50% off)** for backfill captioning and any non-latency-sensitive bulk work.
4. **Local compute = $0**: faster-whisper for all transcription, sentence-transformers for embeddings.

---

## Milestones

### M1 — Skeleton bot + text logging + basic Q&A ✅ DONE

**Scope:** project scaffold (uv, pyproject); SQLite schema (all tables created
up-front: `messages`, `senders`, `media`, `transcripts`, `captions`, `jobs`,
FTS5 `search` with trigram); aiogram bot logging every group message
(media kinds logged as rows with empty content for later milestones);
`/id`, `/stats` commands; @mention/reply handler; Haiku router
(`query/router.py`, structured output, falls back to `context_qa` on error);
Sonnet manual tool loop (`query/agent.py`, max 6 iterations, then forced
final answer with `tool_choice: none`); prompt caching on system+tools;
launchd plist (`deploy/`); tests (`tests/test_store.py`).

**Accepted:** 7/7 tests pass (Russian trigram search, dedup, filters,
transcript-indexed-later searchable); all modules import; tools execute
against real data offline. Live Telegram run pending user's bot token.

### M2 — Voice & video-circle transcription (local, $0) ✅ DONE

**Implemented:** `transcribe.py` — `Transcriber` (lazy faster-whisper
large-v3-turbo, cpu/int8) + `TranscriptionWorker` (dedicated thread with its
own WAL connection — NOT asyncio.to_thread, since sqlite connections are
thread-bound; CTranslate2 releases the GIL). `bot.ingest_media()` downloads
voice/video_note to `media/live/YYYY/MM/`, inserts `media` row, enqueues a
`transcribe` job. Jobs: pending→inflight→done|error, 3 attempts, startup
recovery resets stale inflight. video_note .mp4 decoded by PyAV directly
(no ffmpeg dep). 11 new tests with FakeTranscriber.

**Original scope:**
- Add `faster-whisper` dependency; model `large-v3-turbo` (benchmark vs
  `medium` on the Mac — quality/speed pick is an open question); cache model
  weights under `models/`.
- On incoming `voice` / `video_note`: download via Bot API (≤20MB — voice and
  circles always fit) to `media/live/YYYY/MM/<msgid>.<ext>`; insert `media` row.
- Transcription worker: async queue (don't block the bot loop; whisper runs
  in a thread/process executor), language hint `ru`; write to `transcripts`
  (engine string e.g. `faster-whisper:large-v3-turbo`), index via
  `store.index_text(message_id, transcript)` — **already supported and tested**.
- For `video_note` extract audio with ffmpeg first if needed (faster-whisper
  reads most containers directly via PyAV — verify; ffmpeg fallback).
- Speaker attribution = the message sender (one sender per voice message, no
  diarization needed).
- Oversized media (>20MB regular videos): set `media.skipped=1`, still log the message.
- `jobs` table: enqueue `transcribe` jobs so restarts resume unfinished work.

**Accept:** new voice message transcribed within minutes; transcript findable
via @mention Q&A; $0 API cost for transcription; queue survives restart.

### M3 — History backfill from Telegram Desktop export ✅ DONE

**Implemented:** `backfill/` package — `parser.py` (pure result.json parsing:
text flattening, `from_id` mapping, `date_unixtime`-with-`date`-fallback,
media kinds, the "(File not included...)" sentinel), `media.py` (copies files
to `media/export/YYYY/MM/<msgid><ext>`, sha256 reuse on re-run), `runner.py`
(dedup-at-insert: live row by id wins, fallback `(ts, normalized text)` match;
export media is attached to deduped live rows that lack it — pre-M2 voice
becomes transcribable; FTS indexed only on actual insert → idempotent re-runs;
`BackfillReport.render()`). CLI: `uv run python -m family_assistant.backfill
[export_dir] [--chat-id ID] [--transcribe]` — derives the Bot-API id from the
export id and warns on mismatch; `--transcribe` drains the job queue inline
(resumable), otherwise the bot's worker picks jobs up on next start.
Senders matched on tg user id; export name variants accumulate in
`senders.aliases` without clobbering live display names. Incremental exports
supported: a fuller re-export adds older messages and **upgrades placeholder
media rows in place** (sentinel "(File not included...)" from partial exports,
or skipped >20MB live downloads — export files bypass that cap); real files
are never downgraded. 29 new tests.
**To verify on the real export:** that export `id` == Bot API `message_id`
(check the report's dedup counts over the live-overlap window).

**Original scope:**
- User exports the chat from Telegram Desktop (Settings → Advanced → Export):
  JSON format + media. Place under e.g. `export/` (gitignored).
- `backfill/` module: parse `result.json`; map export senders (`from` /
  `from_id` like `user123456`) to `senders` rows (match on tg user id;
  aliases JSON for name variants).
- Insert with `source='export'`. **Dedup strategy:** in supergroups, export
  `id` generally equals Bot API `message_id` — verify empirically on this
  chat; `UNIQUE(tg_chat_id, tg_message_id, source)` lets both coexist; a
  reconciliation pass prefers `live` rows and ignores matching `export` rows
  (fallback key if ids don't match: `(sender, ts, normalized_text_hash)`).
- Copy/reference media files from export (`media/export/`), insert `media` rows.
- Transcribe all exported voice/video-notes locally: resumable overnight job
  driven by the `jobs` table (expect a few hours for ~3,041 voice messages).
- Index everything into FTS.
- Reconciliation report: counts imported / deduped / skipped / transcribed.

**Accept:** full history queryable ("когда мы первый раз говорили про X");
no duplicates in the live/export overlap window; report printed.

### M4 — Image understanding (Haiku vision + Batch API) ✅ DONE

**Implemented:** `caption.py` — frozen Russian `CAPTION_PROMPT` (byte-stable;
below Haiku's min cacheable prefix so it won't cache yet — expected),
`encode_image` (Pillow, ≤1024px long edge, JPEG q80, original untouched),
`Captioner` (lazy sync Anthropic client) + `CaptionWorker` (thread, same shape
as TranscriptionWorker). **Caption jobs use `ref_id = media.id`** (transcribe
uses `messages.id`). Live photos: `bot.ingest_photo()` downloads
`message.photo[-1]` and the worker captions each immediately with a single
Haiku call (1–2/day — batch discount not worth a scheduler; user-approved).
Backfill: `caption_backfill/` CLI submits Batch API chunks of 500, persists
`jobs.batch_id` → killed runs resume polling instead of re-paying; errored
results retry via fresh batches until `job_max_attempts`; corrupt images
skipped without an API call; `--max-batches N` for a cost-capped trial;
`--retry-errored` revives jobs that exhausted `job_max_attempts` (transient
API failures retry, corrupt images re-skip).
`reset_stale_jobs(unbatched_only=True)` keeps the live worker from stealing
batch-inflight jobs. Captions surface as the message body in all retrieval
queries (`COALESCE(text, transcript, caption)` + «[фото]» prefix). 14 new tests.
**To verify on real data:** reconcile actual batch token spend (< $5 target)
after the first `--max-batches 1` run.

**Original scope:**
- Caption prompt (Russian, one frozen prompt for caching): short factual
  description + visible text + people/objects/place; output ~1–3 sentences.
- Backfill: `jobs` of type `caption` for ~2,174 export photos → **Batch API**
  (50% off; batches of up to a few thousand; poll, write `captions`, index
  into FTS via `index_text`). Estimated cost < $5 total.
- Live photos: download (≤20MB fine), enqueue caption job; process nightly
  batch or on-demand (if a query mentions photos and captions are pending).
- Resize images client-side to ~1024px long edge before sending (token cost ↓).

**Accept:** "найди фото с дачи" returns relevant photo messages (by date/sender
so user can scroll to them in Telegram); backfill cost reconciled < $5.

### M5 — LLM Wiki layer + nightly digest ✅ DONE

**Implemented:** `wiki.py` — `Wiki` (path-validated Layer 2 I/O under `wiki_dir`:
`people/`, `topics/`, `log.md`; rejects traversal / non-`.md` / unknown dirs;
atomic `write_page`; **auto-generated `index.md`** from each page's first line,
so the catalogue is deterministic and not hand-edited by the LLM). The digest
**watermark lives in `log.md`** (last `## YYYY-MM-DD` heading) — no extra table,
wiki stays self-describing and rebuildable. `schema/wiki_guide.md` is the
human-edited Layer 3 (in git; gitignored `wiki/` is Layer 2). `digest/runner.py`
— `DigestAgent` (sync Haiku tool-loop: `list_wiki_index` / `read_wiki_page` /
`write_wiki_page`, bounded by `max_digest_iterations` then a forced no-tools
final = the journal entry) + `run_digest`. Cache layout per the doc: system =
guide (frozen) → start-of-period `index.md` snapshot (breakpoint here) →
period's messages in the user turn; cache hits land on turns 2..N within a
period. Two modes: daily catch-up (watermark+1 → yesterday; fresh wiki digests
only yesterday) and `--rebuild` (wipe Layer 2, **sequential monthly** passes
oldest→newest so each month builds on prior pages — not Batch API, the
cross-month dependency precludes batching). CLI:
`python -m family_assistant.digest [--date YYYY-MM-DD] [--rebuild]`. Query loop
gained `list_wiki_index` / `read_wiki_page` tools (one-time agent prompt-cache
invalidation, expected). `store.messages_between` is the day window
(`COALESCE(text, transcript, caption)`). launchd `deploy/com.family.tgdigest.plist`
(`StartCalendarInterval` 03:00, one-shot, catch-up self-heals missed nights).
21 new tests. **To verify on real data:** `cache_read_input_tokens > 0` on
digest turns 2..N after the first `--rebuild` run.

**Original scope:**
- Directory layout: `wiki/index.md` (catalog of pages, one-line summaries),
  `wiki/log.md` (append-only daily digest journal), `wiki/people/<name>.md`,
  `wiki/topics/<topic>.md`. Fully rebuildable from Layer 1.
- `schema/wiki_guide.md` (Layer 3, human-edited): page naming, create-vs-merge
  rules, format, Russian language, what NOT to store (no secrets/credentials).
- Nightly digest job (separate entrypoint, launchd timer ~03:00 + `pmset`
  wake note in docs): one cache-warmed Haiku pass over the day's messages
  (+ transcripts/captions) → updates relevant wiki pages, appends a dated
  summary to `log.md`, updates `index.md`. Prompt order for caching:
  wiki_guide (frozen) → index.md → day's content.
- Query loop gains tools: `list_wiki_index()`, `read_wiki_page(path)` —
  path-validated to the wiki dir.
- Include `index.md` as ambient context in the agent system block? No — keep
  system frozen; provide via tools (cache stays valid).
- Initial wiki seed: batched Haiku pass over monthly chunks of the
  backfilled history (Batch API), oldest → newest, building pages incrementally.

**Accept:** "что обсуждали на прошлой неделе" answered from `log.md`/topic
pages; one digest call/day with verified cache hits
(`cache_read_input_tokens > 0`); wiki regenerates from scratch deterministically.

### M6 — Semantic search (local embeddings, $0) ✅ DONE

**Implemented:** `embed.py` — `chunk_text` (whitespace-boundary split at
`embedding_chunk_chars`≈512 tokens; short messages → one chunk), `Embedder`
(lazy `sentence-transformers` `intfloat/multilingual-e5-small`, 384-dim, e5
`query: `/`passage: ` prefixes, L2-normalized so cosine == dot product; weights
cache under `models_dir`) + `EmbeddingWorker` (dedicated thread + own WAL
connection, same shape as TranscriptionWorker; `embed` jobs use
`ref_id = messages.id`). New `embeddings` table (one row per chunk, vector as
float32 BLOB, `UNIQUE(message_id, chunk_index, model)` so a model change
re-embeds without collision) — added to `schema.sql`, auto-created on startup
(no migration). Store gained `text_for_message`, `replace_embeddings`
(idempotent — DELETE-then-insert), `messages_needing_embedding` and `knn`
(brute-force numpy cosine kNN: load candidate vectors with optional sender/time
filters → single matmul → dedup chunks to best per message → top-k MessageRows,
`format_rows`-identical to `fts_search`). `semantic_search` tool appended to the
agent's `TOOLS` (one-time prompt-cache invalidation, expected) with frozen
AGENT_SYSTEM guidance (fts first for exact words, semantic for paraphrase/concept
or when fts comes up empty); one shared `Embedder` is threaded from `BotApp`
through `QueryEngine` into the agent and the worker (model loaded once). Live
ingest enqueues `embed` jobs at the three `index_text` chokepoints: text messages
in `bot.log_message`, and after a transcript/caption via an injected
`embed_enqueue` hook on the transcription/caption workers; startup `recover`
self-heals. CLI `embed_backfill/` (`python -m family_assistant.embed_backfill
[--limit N] [--retry-errored]`) — resumable inline drain ($0, no Batch API; the
cross-model/per-message dedup makes re-runs skip done work). 16 new tests with a
`FakeEmbedder` (no real model in CI). **Verified on real data:** real e5 ranks a
paraphrase query ("вопросы здоровья" → "болит голова"/"к врачу") above unrelated
messages; `--limit 50` trial embedded 50/50, 0 errored, 384-dim/1536-byte vectors.

**Original scope:**
- `sentence-transformers` + `intfloat/multilingual-e5-small` (good Russian,
  small, fast on M-series CPU). Remember e5 conventions: prefix `query: ` /
  `passage: ` on encode.
- Embed messages, transcripts, captions (chunk long texts ~512 tokens);
  store vectors in `sqlite-vec` virtual table (or a flat numpy index — at
  this scale, <100k rows, either works).
- Backfill embeddings job (resumable via `jobs`); embed new content on ingest.
- New agent tool: `semantic_search(query, k)` → kNN over vectors, returns
  same message format as `fts_search`.
- Update agent system prompt guidance: try fts first for exact words, semantic
  for paraphrase ("когда мы говорили о здоровье" finds "болит голова").
  Note: changing the system prompt/tool list invalidates the prompt cache
  once — expected, one-time.

**Accept:** paraphrase queries that FTS misses return correct results;
$0 embedding cost; ingest latency still fine.

### M7 — Video understanding + polish

**Scope:**
- Video ingest: ffmpeg extract audio → whisper transcript; sample 1–3
  keyframes → Haiku caption; both indexed. Skip >20MB live videos (already
  flagged); backfilled export videos are local so size is no issue.
- Scheduled weekly summary: Sunday evening digest posted to the chat
  (opt-in; reuse digest machinery from M5).
- `/commands`: `/summary [period]`, `/find <query>`, `/wiki <topic>`, `/spend`.
- Token-spend metrics: accumulate `usage` from every API call into a
  `spend(date, model, in_tokens, cached_tokens, out_tokens)` table; `/spend`
  reports the current month estimate.
- Better routing: thresholds, maybe skip the router for obvious cases
  (starts with "найди" → search_history).

**Accept:** videos searchable by spoken content; weekly auto-summary posts;
monthly spend visible and within ~$3–10.

---

## Operational notes / gotchas (learned or anticipated)

- **Mac sleep:** Telegram long-polling backlog is held ~24h; gaps longer than
  that lose live messages (rare; acceptable — recoverable by re-running the
  M3 backfill with a fresh export, which fills gaps incrementally). launchd
  `KeepAlive` restarts the bot; digest needs a `pmset repeat wakeorpoweron`
  schedule or runs on next wake. On wake the backlog is archived normally,
  but @mentions older than `ANSWER_MAX_AGE_MINUTES` (default 30) are not
  answered (no ghost replies / Sonnet burst); startup logs a warning when the
  archive gap exceeds 24h, and `/stats` shows the newest message time.
- **Privacy mode** must stay disabled; if the bot is re-added or BotFather
  settings change, group messages silently stop arriving — `/stats` is the
  quick check (count should grow).
- **Prompt-cache hygiene:** system prompt and tool list must stay byte-stable.
  Volatile data (current date, asker name) goes in the *user* message —
  already done in M1 (`agent.py`).
- **Anthropic minimum cacheable prefix** (Sonnet 4.6: 2048 tokens): the M1
  system+tools prefix may be below it (silent no-cache, harmless). It will
  start paying off when the prefix grows in M5.
- **Trigram FTS needs ≥3-char queries**; `store.search` returns `[]` on
  malformed/short queries instead of raising (tested).
- **Export sender ids**: Desktop export uses `from_id: "user123456"` strings;
  strip the `user` prefix to map to Bot API numeric ids.
- All data stays local except retrieval slices/images sent to Anthropic for
  answering/captioning (user-accepted tradeoff).

## Current file map (M6)

```
src/family_assistant/
├── bot.py            # aiogram handlers; BotApp; logging + @mention answering
├── config.py         # pydantic-settings; .env; models, paths, limits
├── __main__.py       # python -m family_assistant
├── transcribe.py     # M2: Transcriber (faster-whisper) + TranscriptionWorker
├── caption.py        # M4: CAPTION_PROMPT, encode_image, Captioner + CaptionWorker
├── caption_backfill/ # M4: Batch API photo captioning CLI (resumable)
│   ├── runner.py     # enqueue → submit chunks → poll/collect; jobs.batch_id resume
│   └── __main__.py   # CLI: python -m family_assistant.caption_backfill [--max-batches N] [--retry-errored]
├── embed.py          # M6: chunk_text, Embedder (e5, sentence-transformers) + EmbeddingWorker
├── embed_backfill/   # M6: local embedding backfill CLI (resumable, $0)
│   ├── runner.py     # enqueue → inline drain; report; per-model/message dedup
│   └── __main__.py   # CLI: python -m family_assistant.embed_backfill [--limit N] [--retry-errored]
├── wiki.py           # M5: Wiki — Layer 2 I/O, path validation, auto-index, log watermark
├── digest/           # M5: nightly wiki digest
│   ├── runner.py     # DigestAgent (Haiku tool-loop) + run_digest (daily / monthly seed)
│   └── __main__.py   # CLI: python -m family_assistant.digest [--date YYYY-MM-DD] [--rebuild]
├── store/
│   ├── schema.sql    # full schema incl. M2-M6 tables (media/transcripts/captions/jobs/embeddings)
│   └── db.py         # Store: upsert_sender, insert_message, index_text, search,
│                     #        get_messages_around, recent_window, messages_between, stats,
│                     #        text_for_message, replace_embeddings, messages_needing_embedding, knn
├── query/
│   ├── router.py     # classify_intent (Haiku, structured output)
│   ├── agent.py      # RetrievalAgent: TOOLS (+wiki +semantic_search), manual loop,
│   │                 #        prompt caching, answer / answer_generic
│   └── engine.py     # QueryEngine.handle: route → answer (shares one Embedder)
└── backfill/         # M3: Telegram Desktop export import
    ├── parser.py     # pure result.json parsing → ParsedMessage
    ├── media.py      # copy export files → media/export/YYYY/MM/, sha256
    ├── runner.py     # dedup-at-insert, media attach, BackfillReport
    └── __main__.py   # CLI: python -m family_assistant.backfill
schema/wiki_guide.md  # M5: Layer 3 — human-edited wiki maintenance rules (in git)
tests/                # store/FTS, transcription, backfill, captioning, wiki, digest, embed (104 tests)
deploy/com.family.tgassistant.plist   # bot (KeepAlive)
deploy/com.family.tgdigest.plist      # M5: nightly digest (StartCalendarInterval 03:00)
```
