"""Digest runner tests with a fake (scripted) Haiku client."""

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from family_assistant.digest import run_digest
from family_assistant.query.agent import RetrievalAgent
from family_assistant.store import Store
from family_assistant.wiki import Wiki


class FakeSettings:
    anthropic_api_key = "test"
    digest_model = "claude-haiku-4-5"
    digest_max_tokens = 2048
    max_digest_iterations = 8
    digest_chat_id = -100
    allowed_chat_ids = [-100]
    wiki_dir = None  # passed explicitly in tests
    wiki_guide_path = None


def _usage(cache_read=0):
    return SimpleNamespace(
        input_tokens=100, cache_read_input_tokens=cache_read, output_tokens=10
    )


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(tid, name, inp):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


class FakeResp:
    def __init__(self, content, stop_reason, cache_read=0):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _usage(cache_read)


class FakeMessages:
    """Returns scripted responses in order; records every create() kwargs."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def fake_client(responses):
    return SimpleNamespace(messages=FakeMessages(responses))


def write_then_finish(path, content, summary):
    """One period's worth of responses: write a page, then a journal summary."""
    return [
        FakeResp([_tool("t1", "write_wiki_page", {"path": path, "content": content})],
                 "tool_use"),
        FakeResp([_text(summary)], "end_turn", cache_read=80),
    ]


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path / "test.db")
    guide = tmp_path / "wiki_guide.md"
    guide.write_text("правила вики", encoding="utf-8")
    wiki = Wiki(tmp_path / "wiki", guide)
    yield store, wiki
    store.close()


def add_msg(store, day: date, text, *, sender="Анна", uid=1, msg_id=None):
    sid = store.upsert_sender(uid, sender)
    ts = int(datetime(day.year, day.month, day.day, 12, 0).timestamp())
    mid = msg_id if msg_id is not None else int(ts) % 1_000_000
    return store.insert_message(
        tg_message_id=mid, tg_chat_id=-100, sender_id=sid, ts=ts,
        kind="text", text=text,
    )


def test_single_day_writes_page_and_log(env):
    store, wiki = env
    add_msg(store, date(2024, 5, 10), "Я вышла на новую работу врачом", msg_id=1)
    client = fake_client(
        write_then_finish("people/anna.md", "# Анна\n\nРаботает врачом.",
                          "Анна рассказала о новой работе.")
    )

    report = run_digest(
        store=store, settings=FakeSettings(), wiki=wiki, client=client,
        only_date=date(2024, 5, 10),
    )

    assert report.periods_processed == 1
    assert report.pages_written == 1
    assert wiki.read_page("people/anna.md") == "# Анна\n\nРаботает врачом.\n"
    log = wiki.read_page("log.md")
    assert "## 2024-05-10" in log and "Анна рассказала о новой работе." in log
    assert "people/anna.md" in wiki.read_index()
    # cache breakpoint sits on the index block (guide+index cached together)
    system = client.messages.calls[0]["system"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}


def test_empty_day_logs_placeholder_without_api(env):
    store, wiki = env  # no messages
    client = fake_client([])  # any API call would IndexError-pop from empty

    report = run_digest(
        store=store, settings=FakeSettings(), wiki=wiki, client=client,
        only_date=date(2024, 5, 10),
    )

    assert report.periods_empty == 1
    assert report.periods_processed == 0
    assert client.messages.calls == []
    assert "(нет сообщений)" in wiki.read_page("log.md")


def test_watermark_skips_already_digested(env):
    store, wiki = env
    wiki.append_log(date(2024, 5, 10), "уже сделано")
    client = fake_client([])

    # today=2024-05-11 -> yesterday=05-10 == watermark -> nothing to do
    report = run_digest(
        store=store, settings=FakeSettings(), wiki=wiki, client=client,
        today=date(2024, 5, 11),
    )

    assert report.periods_processed == 0
    assert client.messages.calls == []
    assert "already up to date" in report.note


def test_rebuild_seeds_monthly_oldest_first(env):
    store, wiki = env
    add_msg(store, date(2024, 4, 15), "поехали на дачу", msg_id=1)
    add_msg(store, date(2024, 5, 20), "ремонт на даче", msg_id=2)
    # a stray pre-existing page must be wiped by --rebuild
    wiki.write_page("topics/stale.md", "# Старое")

    responses = (
        write_then_finish("topics/dacha.md", "# Дача", "Апрель: поехали на дачу.")
        + write_then_finish("topics/dacha.md", "# Дача\n\nРемонт.", "Май: ремонт.")
    )
    report = run_digest(
        store=store, settings=FakeSettings(), wiki=wiki,
        client=fake_client(responses), today=date(2024, 6, 1), rebuild=True,
    )

    assert report.mode == "monthly"
    assert report.periods_processed == 2
    assert wiki.read_page("topics/stale.md") is None  # reset wiped it
    log = wiki.read_page("log.md")
    assert "## 2024-04-30" in log and "## 2024-05-31" in log
    # April digested before May (oldest -> newest)
    assert log.index("2024-04-30") < log.index("2024-05-31")


def test_iteration_cap_forces_final_answer(env):
    store, wiki = env
    add_msg(store, date(2024, 5, 10), "сообщение", msg_id=1)
    settings = FakeSettings()
    settings.max_digest_iterations = 2
    # Always ask for a tool — never stop — so the cap kicks in.
    loop = [
        FakeResp([_tool(f"t{i}", "list_wiki_index", {})], "tool_use")
        for i in range(2)
    ]
    final = FakeResp([_text("Финальное résumé.")], "end_turn")
    client = fake_client(loop + [final])

    report = run_digest(
        store=store, settings=settings, wiki=wiki, client=client,
        only_date=date(2024, 5, 10),
    )

    assert report.periods_processed == 1
    # 2 loop turns + 1 forced final, and the forced call disables tools
    assert len(client.messages.calls) == 3
    assert client.messages.calls[-1]["tool_choice"] == {"type": "none"}
    assert "Финальное résumé." in wiki.read_page("log.md")


def test_retrieval_agent_wiki_tools(env):
    store, wiki = env
    wiki.write_page("topics/dacha.md", "# Дача\n\nВ Тверской области.")
    agent = RetrievalAgent(client=None, settings=FakeSettings(), store=store, wiki=wiki)

    index = agent._run_tool("list_wiki_index", {}, -100)
    assert "topics/dacha.md" in index
    page = agent._run_tool("read_wiki_page", {"path": "topics/dacha.md"}, -100)
    assert "Тверской области" in page
    missing = agent._run_tool("read_wiki_page", {"path": "people/none.md"}, -100)
    assert missing == "(страница не существует)"
    bad = agent._run_tool("read_wiki_page", {"path": "../evil.md"}, -100)
    assert bad.startswith("Ошибка:")
