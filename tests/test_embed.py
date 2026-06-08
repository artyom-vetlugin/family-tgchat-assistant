"""M6 semantic-search tests — Store kNN, worker and backfill with a fake
embedder (no real sentence-transformers model loaded)."""

import time

import numpy as np
import pytest

from family_assistant.embed import EmbeddingWorker, chunk_text
from family_assistant.embed_backfill import run_embed_backfill
from family_assistant.store import Store


class FakeEmbedder:
    """Deterministic unit vectors: chunk i → one-hot at axis (i % 4)."""

    model_name = "fake-embed"
    dim = 4

    def __init__(self, fail_times=0):
        self.fail_times = fail_times
        self.calls = 0

    def encode_passages(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("encode failed")
        out = []
        for i, _t in enumerate(texts):
            v = np.zeros(4, dtype=np.float32)
            v[i % 4] = 1.0
            out.append(v)
        return np.array(out, dtype=np.float32)

    def encode_query(self, text):
        v = np.zeros(4, dtype=np.float32)
        v[0] = 1.0
        return v


class FakeSettings:
    job_max_attempts = 3
    embedding_chunk_chars = 1800
    embedding_model = "fake-embed"
    embedding_dim = 4

    def __init__(self, db_path):
        self.db_path = db_path


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


def unit(v):
    v = np.array(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def text_msg(store, mid, text, *, ts=None, name="Папа", uid=1):
    sid = store.upsert_sender(uid, name)
    rid = store.insert_message(
        tg_message_id=mid, tg_chat_id=-100, sender_id=sid,
        ts=ts or int(time.time()), kind="text", text=text,
    )
    store.index_text(rid, text)
    return rid


def embed(store, rid, vectors, model="m"):
    store.replace_embeddings(rid, np.array(vectors, dtype=np.float32), model=model, dim=4)


# --- chunk_text ----------------------------------------------------------


def test_chunk_text_short_is_one_chunk():
    assert chunk_text("привет", 1800) == ["привет"]


def test_chunk_text_empty():
    assert chunk_text("   ", 1800) == []


def test_chunk_text_splits_long_on_whitespace():
    long = " ".join(["слово"] * 500)
    chunks = chunk_text(long, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # no words lost across the split
    assert " ".join(chunks).split() == long.split()


def test_chunk_text_keeps_oversized_word_whole():
    assert chunk_text("a" * 50, 10) == ["a" * 50]


# --- Store.knn / embeddings ---------------------------------------------


def test_knn_orders_by_cosine(store):
    a = text_msg(store, 1, "дача")
    b = text_msg(store, 2, "здоровье")
    embed(store, a, [[1, 0, 0, 0]])
    embed(store, b, [[0, 1, 0, 0]])
    rows = store.knn(unit([0.9, 0.1, 0, 0]), model="m", k=2)
    assert [r.tg_message_id for r in rows] == [1, 2]


def test_knn_dedups_chunks_to_best(store):
    c = text_msg(store, 3, "длинный текст")
    embed(store, c, [[0, 0, 1, 0], [0, 0, 0, 1]])
    rows = store.knn(unit([0, 0, 0, 1]), model="m", k=5)
    ids = [r.tg_message_id for r in rows]
    assert ids.count(3) == 1  # surfaced once despite two chunks


def test_knn_filters_by_sender_and_time(store):
    old = text_msg(store, 10, "x", ts=1000, name="Аня", uid=2)
    new = text_msg(store, 11, "y", ts=2000, name="Боб", uid=3)
    embed(store, old, [[1, 0, 0, 0]])
    embed(store, new, [[1, 0, 0, 0]])
    q = unit([1, 0, 0, 0])
    assert [r.tg_message_id for r in store.knn(q, model="m", k=5, sender="Аня")] == [10]
    assert [r.tg_message_id for r in store.knn(q, model="m", k=5, after_ts=1500)] == [11]
    assert [r.tg_message_id for r in store.knn(q, model="m", k=5, before_ts=1500)] == [10]


def test_knn_empty_and_model_scoped(store):
    a = text_msg(store, 1, "дача")
    embed(store, a, [[1, 0, 0, 0]], model="m")
    assert store.knn(unit([1, 0, 0, 0]), model="m", k=5)  # found
    assert store.knn(unit([1, 0, 0, 0]), model="other", k=5) == []  # different model


def test_replace_embeddings_is_idempotent(store):
    a = text_msg(store, 1, "дача")
    embed(store, a, [[1, 0, 0, 0], [0, 1, 0, 0]])
    embed(store, a, [[0, 0, 1, 0]])  # re-embed replaces
    n = store.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE message_id=?", (a,)
    ).fetchone()[0]
    assert n == 1


def test_text_for_message_coalesce(store):
    rid = text_msg(store, 1, "текст")
    assert store.text_for_message(rid) == "текст"


def test_messages_needing_embedding(store):
    a = text_msg(store, 1, "hello")
    # voice + transcript
    sid = store.upsert_sender(1, "Папа")
    v = store.insert_message(
        tg_message_id=2, tg_chat_id=-100, sender_id=sid,
        ts=int(time.time()), kind="voice", text=None,
    )
    store.insert_transcript(message_id=v, text="расшифровка", engine="e")
    # photo + caption
    p = store.insert_message(
        tg_message_id=3, tg_chat_id=-100, sender_id=sid,
        ts=int(time.time()), kind="photo", text=None,
    )
    media = store.insert_media(message_id=p, kind="photo", rel_path="x.jpg")
    store.insert_caption(media_id=media, text="описание", model="m")
    # textless sticker
    s = store.insert_message(
        tg_message_id=4, tg_chat_id=-100, sender_id=sid,
        ts=int(time.time()), kind="sticker", text=None,
    )

    need = store.messages_needing_embedding("m")
    assert a in need and v in need and p in need
    assert s not in need  # no searchable text

    embed(store, a, [[1, 0, 0, 0]], model="m")
    assert a not in store.messages_needing_embedding("m")
    assert a in store.messages_needing_embedding("other")  # per-model


# --- EmbeddingWorker -----------------------------------------------------


def test_worker_embeds_message(tmp_path, store):
    settings = FakeSettings(tmp_path / "test.db")  # same db file as the fixture
    rid = text_msg(store, 1, "привет")
    job = store.create_job(job_type="embed", ref_id=rid)
    worker = EmbeddingWorker(settings.db_path, settings, FakeEmbedder())
    worker._process(store, job)
    assert store.job_state(job)[0] == "done"
    n = store.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE message_id=?", (rid,)
    ).fetchone()[0]
    assert n == 1


def test_worker_retries_then_succeeds(tmp_path, store):
    settings = FakeSettings(tmp_path / "test.db")
    rid = text_msg(store, 1, "привет")
    job = store.create_job(job_type="embed", ref_id=rid)
    worker = EmbeddingWorker(settings.db_path, settings, FakeEmbedder(fail_times=1))
    worker._process(store, job)
    assert store.job_state(job)[0] == "pending"  # first attempt failed → retry
    worker._process(store, job)
    assert store.job_state(job)[0] == "done"


def test_worker_recover_resets_stale(tmp_path, store):
    settings = FakeSettings(tmp_path / "test.db")
    rid = text_msg(store, 1, "привет")
    job = store.create_job(job_type="embed", ref_id=rid)
    store.claim_job(job)  # left inflight by a "crash"
    worker = EmbeddingWorker(settings.db_path, settings, FakeEmbedder())
    worker.recover(store)
    assert store.job_state(job)[0] == "pending"
    assert job in list(worker.queue.queue)


# --- backfill ------------------------------------------------------------


def test_backfill_embeds_and_is_resumable(tmp_path, store):
    settings = FakeSettings(tmp_path / "test.db")
    text_msg(store, 1, "a")
    text_msg(store, 2, "b")
    report = run_embed_backfill(store=store, settings=settings, embedder=FakeEmbedder())
    assert report.enqueued == 2
    assert report.done == 2
    assert report.chunks == 2
    assert report.pending_left == 0

    again = run_embed_backfill(store=store, settings=settings, embedder=FakeEmbedder())
    assert again.enqueued == 0  # nothing left to do


def test_backfill_limit(tmp_path, store):
    settings = FakeSettings(tmp_path / "test.db")
    for i in range(1, 6):
        text_msg(store, i, f"msg {i}")
    report = run_embed_backfill(
        store=store, settings=settings, embedder=FakeEmbedder(), limit=2
    )
    assert report.enqueued == 2
    assert len(store.messages_needing_embedding("fake-embed")) == 3
