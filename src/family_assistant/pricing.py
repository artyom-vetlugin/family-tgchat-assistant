"""Token pricing for the `/spend` estimate (M7).

These prices are **approximate** and change over time — they live here, in one
obvious place, so `/spend` can turn the `spend` table's token counts into a
rough dollar figure. Always treat the result as an estimate (the command says
so), and update PRICES when Anthropic's pricing changes.

Prices are USD per million tokens. Cache reads bill at ~0.1x input; 5-minute
cache writes at ~1.25x input. Batch API spend is recorded under a `<model>:batch`
key (50% off) so it estimates at half price without a separate table.

Source: docs.anthropic.com/pricing (Haiku 4.5 $1/$5 in/out, Sonnet 4.6 $3/$15).
"""

from __future__ import annotations

# Per-million-token USD prices. Keys are the exact model strings written to the
# `spend` table. Update when Anthropic changes pricing.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"in": 1.00, "cached": 0.10, "cache_write": 1.25, "out": 5.00},
    "claude-sonnet-4-6": {"in": 3.00, "cached": 0.30, "cache_write": 3.75, "out": 15.00},
}

_MILLION = 1_000_000


def _row_cost(row: dict, price: dict[str, float]) -> float:
    return (
        row.get("in_tokens", 0) * price["in"]
        + row.get("cached_tokens", 0) * price["cached"]
        + row.get("cache_write_tokens", 0) * price["cache_write"]
        + row.get("out_tokens", 0) * price["out"]
    ) / _MILLION


def estimate_usd(rows: list[dict]) -> tuple[float, list[str]]:
    """Estimate total USD for `spend_summary` rows.

    Returns (total_usd, unknown_models) — a model with no price entry is added
    to `unknown_models` and contributes $0, so the caller can flag it rather
    than silently under-reporting. A `<model>:batch` key prices at half its base
    model's rate (Batch API discount)."""
    total = 0.0
    unknown: list[str] = []
    for row in rows:
        model = row["model"]
        base, is_batch = (model[:-6], True) if model.endswith(":batch") else (model, False)
        price = PRICES.get(base)
        if price is None:
            unknown.append(model)
            continue
        cost = _row_cost(row, price)
        total += cost * 0.5 if is_batch else cost
    return total, unknown
