"""Pricing estimate tests (M7) — no API calls."""

from family_assistant import pricing


def test_estimate_basic_haiku():
    rows = [
        {
            "model": "claude-haiku-4-5",
            "in_tokens": 1_000_000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "out_tokens": 1_000_000,
        }
    ]
    total, unknown = pricing.estimate_usd(rows)
    assert unknown == []
    # 1M input @ $1 + 1M output @ $5 = $6.00
    assert round(total, 4) == 6.00


def test_cached_and_cache_write_priced_separately():
    rows = [
        {
            "model": "claude-sonnet-4-6",
            "in_tokens": 0,
            "cached_tokens": 1_000_000,      # $0.30
            "cache_write_tokens": 1_000_000,  # $3.75
            "out_tokens": 0,
        }
    ]
    total, _ = pricing.estimate_usd(rows)
    assert round(total, 4) == 4.05


def test_batch_suffix_prices_at_half():
    rows = [
        {
            "model": "claude-haiku-4-5:batch",
            "in_tokens": 1_000_000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "out_tokens": 0,
        }
    ]
    total, unknown = pricing.estimate_usd(rows)
    assert unknown == []
    assert round(total, 4) == 0.50  # half of $1.00


def test_unknown_model_flagged_not_silent():
    rows = [
        {
            "model": "some-future-model",
            "in_tokens": 1_000_000,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
            "out_tokens": 0,
        }
    ]
    total, unknown = pricing.estimate_usd(rows)
    assert total == 0.0
    assert unknown == ["some-future-model"]
