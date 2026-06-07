"""Pure-function tests for backlog catch-up behavior (no aiogram fixtures)."""

from family_assistant.bot import archive_gap_warning, too_old_to_answer

NOW = 1_750_000_000


def test_fresh_mention_is_answered():
    assert not too_old_to_answer(NOW - 5 * 60, NOW, max_age_minutes=30)


def test_stale_mention_is_skipped():
    assert too_old_to_answer(NOW - 3 * 3600, NOW, max_age_minutes=30)


def test_staleness_boundary():
    assert not too_old_to_answer(NOW - 30 * 60, NOW, max_age_minutes=30)  # exactly at limit
    assert too_old_to_answer(NOW - 30 * 60 - 1, NOW, max_age_minutes=30)


def test_zero_max_age_disables_skipping():
    assert not too_old_to_answer(NOW - 100 * 3600, NOW, max_age_minutes=0)


def test_no_gap_warning_when_recent():
    assert archive_gap_warning(NOW - 3600, NOW) is None
    assert archive_gap_warning(NOW - 24 * 3600, NOW) is None  # exactly 24h — still held


def test_gap_warning_after_24h():
    warning = archive_gap_warning(NOW - 30 * 3600, NOW)
    assert warning is not None
    assert "30h" in warning
    assert "backfill" in warning


def test_no_gap_warning_for_empty_archive():
    assert archive_gap_warning(None, NOW) is None
