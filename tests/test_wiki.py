"""Wiki (Layer 2) file I/O, path validation, index generation, watermark."""

from datetime import date

import pytest

from family_assistant.wiki import Wiki


@pytest.fixture
def wiki(tmp_path):
    guide = tmp_path / "wiki_guide.md"
    guide.write_text("правила", encoding="utf-8")
    return Wiki(tmp_path / "wiki", guide)


def test_write_read_roundtrip(wiki):
    wiki.write_page("people/anna.md", "# Анна\n\nЛюбит дачу.")
    assert wiki.read_page("people/anna.md") == "# Анна\n\nЛюбит дачу.\n"
    assert wiki.read_page("people/missing.md") is None


@pytest.mark.parametrize(
    "bad",
    [
        "../evil.md",            # traversal
        "people/../../evil.md",  # traversal through allowed dir
        "secrets.md",            # root file not in allowlist
        "other/x.md",            # unknown subdir
        "people/anna.txt",       # not markdown
        "people/sub/anna.md",    # too deep
    ],
)
def test_rejects_unsafe_paths(wiki, bad):
    with pytest.raises(ValueError):
        wiki.write_page(bad, "x")
    with pytest.raises(ValueError):
        wiki.read_page(bad)


def test_list_pages_sorted(wiki):
    wiki.write_page("topics/dacha.md", "# Дача")
    wiki.write_page("people/anna.md", "# Анна")
    wiki.write_page("people/boris.md", "# Борис")
    assert wiki.list_pages() == ["people/anna.md", "people/boris.md", "topics/dacha.md"]


def test_index_lists_pages_with_summaries(wiki):
    wiki.write_page("people/anna.md", "# Анна — старшая дочь\n\nРаботает врачом.")
    wiki.write_page("topics/dacha.md", "# Дача в Тверской области")
    index = wiki.regenerate_index()
    assert "people/anna.md — Анна — старшая дочь" in index
    assert "topics/dacha.md — Дача в Тверской области" in index
    # auto-generated index is what read_index returns
    assert wiki.read_index() == index


def test_empty_index_has_placeholders(wiki):
    index = wiki.render_index()
    assert "## Люди" in index and "## Темы" in index
    assert "(пусто)" in index


def test_append_log_and_watermark(wiki):
    assert wiki.last_digested_date() is None
    wiki.append_log(date(2024, 5, 10), "Обсуждали дачу.")
    wiki.append_log(date(2024, 5, 11), "(нет сообщений)")
    log = wiki.read_page("log.md")
    assert "# Журнал" in log
    assert "## 2024-05-10" in log and "Обсуждали дачу." in log
    assert "## 2024-05-11" in log
    assert wiki.last_digested_date() == date(2024, 5, 11)


def test_reset_wipes_layer2_not_guide(wiki):
    wiki.write_page("people/anna.md", "# Анна")
    wiki.append_log(date(2024, 5, 10), "x")
    wiki.reset()
    assert wiki.list_pages() == []
    assert wiki.read_page("log.md") is None
    assert wiki.guide() == "правила"  # Layer 3 untouched
