import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from evaluation.pricing import parse_prices, select_rate
from evaluation.token_cost import price_normalized_usage
from evaluation.token_normalizers import normalize_provider_usage


@pytest.mark.asyncio
async def test_refresh_persists_snapshot_and_retains_it_when_source_fails(monkeypatch, tmp_path):
    from evaluation import pricing
    monkeypatch.delenv("EVALUATION_PRICE_SNAPSHOT_PATH", raising=False)
    monkeypatch.setenv("EVALUATION_PRICE_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(pricing, "_snapshot", None)
    monkeypatch.setattr(pricing, "_error", None)
    client = AsyncMock()
    response = MagicMock(text=HTML)
    response.raise_for_status.return_value = None
    client.get.return_value = response
    client.__aenter__.return_value = client
    monkeypatch.setattr(pricing.httpx, "AsyncClient", lambda **kwargs: client)
    first = await pricing.refresh_prices()
    assert first.status == "ready"
    assert (tmp_path / f"{first.snapshot_id}.json").exists()
    client.get.side_effect = httpx.ConnectError("offline")
    failed = await pricing.refresh_prices()
    assert failed.status == "sync_failed"
    assert failed.snapshot_id == first.snapshot_id


HTML = """<div><h2 id="gemini-3.8-flash">Gemini 3.8 Flash</h2></div>
<devsite-selector><section><h3>Flex</h3><table><tbody>
<tr><td>Input price</td><td>Free</td><td>$0.375 through December 31, 2026.<br>$0.75 starting January 1, 2027.</td></tr>
<tr><td>Output price (including thinking tokens)</td><td>Free</td><td>$1.875 through December 31, 2026.<br>$3.75 starting January 1, 2027.</td></tr>
<tr><td>Context caching price</td><td>Free</td><td>$0.0375 through December 31, 2026.<br>$0.075 starting January 1, 2027.<br>$0.50 / 1,000,000 tokens per hour (storage price)</td></tr>
</tbody></table></section></devsite-selector>"""


def test_price_date_cache_and_thinking_are_not_double_charged():
    snapshot = {**parse_prices(HTML), "snapshot_id": "official-test"}
    usage = normalize_provider_usage(
        "google",
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "output_token_details": {"reasoning": 5},
            "input_token_details": {"cache_read": 80},
            "total_tokens": 120,
        },
    )
    assert usage.output_text_tokens == 15
    assert usage.cached_input_tokens == 80
    old = price_normalized_usage(
        "gemini-3.8-flash",
        usage,
        snapshot,
        service_tier="flex",
        created_at="2026-09-17",
    )
    new = price_normalized_usage(
        "gemini-3.8-flash",
        usage,
        snapshot,
        service_tier="flex",
        created_at="2027-01-01",
    )
    assert old["estimated_cost_usd"] == pytest.approx(
        (20 * 0.375 + 80 * 0.0375 + 20 * 1.875) / 1e6
    )
    assert new["estimated_cost_usd"] == pytest.approx(old["estimated_cost_usd"] * 2)


def test_unknown_tier_does_not_use_standard_price():
    snapshot = {**parse_prices(HTML), "snapshot_id": "official-test"}
    usage = normalize_provider_usage(
        "google", {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
    )
    assert (
        price_normalized_usage("gemini-3.8-flash", usage, snapshot)[
            "estimated_cost_usd"
        ]
        is None
    )


def test_unknown_units_and_overlapping_tiers_are_not_guessed():
    with pytest.raises(ValueError):
        parse_prices(HTML.replace("$0.375 through", "$0.375 per minute through"))
    assert (
        select_rate(
            [{"usd_per_1m": 1}, {"usd_per_1m": 2}], input_tokens=10, day="2026-09-17"
        )
        is None
    )


def test_context_threshold_and_missing_cache_are_distinct_from_zero():
    records = [
        {"usd_per_1m": 1, "max_input": 200000},
        {"usd_per_1m": 2, "min_input_exclusive": 200000},
    ]
    assert select_rate(records, input_tokens=200000, day="2026-09-17") == 1
    assert select_rate(records, input_tokens=200001, day="2026-09-17") == 2
    assert (
        normalize_provider_usage(
            "google", {"input_tokens": 10, "total_tokens": 10}
        ).cached_input_tokens
        is None
    )
