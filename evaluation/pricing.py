"""Refresh public Gemini text prices; preserve immutable estimation snapshots."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
logger = logging.getLogger(__name__)
_snapshot: dict[str, Any] | None = None
_error: str | None = None
_lock = asyncio.Lock()


class PricingStatus(BaseModel):
    snapshot_id: str | None = None
    fetched_at: str | None = None
    source: str = SOURCE
    model_count: int = 0
    status: str = "unavailable"
    evaluator_model: str = "gemini-3.1-flash-lite"


def _directory() -> Path:
    return Path(os.getenv("EVALUATION_PRICE_DIRECTORY", "output/evaluation-prices"))


def current_snapshot() -> dict[str, Any]:
    global _snapshot
    if _snapshot is None:
        try:
            _snapshot = json.loads(
                (_directory() / "current.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {"snapshot_id": "local-unknown", "currency": "USD", "models": {}}
    return _snapshot


def pricing_status() -> PricingStatus:
    if os.getenv("EVALUATION_PRICE_SNAPSHOT_PATH"):
        from evaluation.token_cost import load_price_snapshot

        try:
            snapshot = load_price_snapshot()
            return PricingStatus(
                snapshot_id=snapshot["snapshot_id"],
                model_count=len(snapshot["models"]),
                source="EVALUATION_PRICE_SNAPSHOT_PATH",
                status="manual",
                evaluator_model=os.getenv(
                    "EVALUATION_EVALUATOR_MODEL", "gemini-3.1-flash-lite"
                ),
            )
        except ValueError:
            return PricingStatus(status="invalid_manual_price")
    snapshot = current_snapshot()
    return PricingStatus(
        snapshot_id=snapshot.get("snapshot_id"),
        fetched_at=snapshot.get("fetched_at"),
        model_count=len(snapshot.get("models", {})),
        status="sync_failed"
        if _error
        else ("ready" if snapshot.get("models") else "unavailable"),
        evaluator_model=os.getenv(
            "EVALUATION_EVALUATOR_MODEL", "gemini-3.1-flash-lite"
        ),
    )


def _rates(cell: Any) -> list[dict[str, Any]]:
    rates = []
    for line in cell.get_text("\n", strip=True).splitlines():
        if "storage" in line or "audio" in line and "text" not in line:
            continue
        match = re.match(r"^\$([0-9]+(?:\.[0-9]+)?)\b(.*)$", line)
        if not match:
            continue
        suffix = match[2].strip()
        rate: dict[str, Any] = {"usd_per_1m": float(match[1])}
        for keyword, key in (("through", "valid_until"), ("starting", "valid_from")):
            date = re.search(keyword + r" ([A-Za-z]+ \d{1,2}, \d{4})", suffix)
            if date:
                rate[key] = datetime.strptime(date[1], "%B %d, %Y").date().isoformat()
                suffix = suffix.replace(date[0], "")
        threshold = re.search(r"prompts\s*(<=|≤|>)\s*(\d+)k(?: tokens)?", suffix)
        if threshold:
            rate[
                "max_input" if threshold[1] in {"<=", "≤"} else "min_input_exclusive"
            ] = int(threshold[2]) * 1000
            suffix = suffix.replace(threshold[0], "")
        suffix = re.sub(r"\((?:text|image|video|audio|\s|/)+\)", "", suffix)
        if suffix.strip(" .,\n"):
            # Never guess a newly introduced modality, unit, or condition.
            raise ValueError("Unrecognized official pricing condition")
        rates.append(rate)
    if not rates:
        raise ValueError("No supported text price")
    return rates


def parse_prices(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    models: dict[str, Any] = {}
    for heading in soup.find_all("h2"):
        model = str(heading.get("id", ""))
        if not re.fullmatch(
            r"gemini-(?:2\.5|3(?:\.\d+)?)-(?:flash(?:-lite)?|pro)(?:-preview(?:-[\d-]+)?)?",
            model,
        ):
            continue
        tiers: dict[str, Any] = {}
        for element in heading.find_all_next(["h2", "table"]):
            if element.name == "h2":
                break
            section = element.find_parent("section")
            title = section.find("h3") if section else None
            tier = title.get_text(strip=True).lower() if title else "standard"
            if tier not in {"standard", "flex"}:
                continue
            prices: dict[str, Any] = {}
            try:
                for row in element.select("tbody tr"):
                    cells = row.find_all("td", recursive=False)
                    if len(cells) != 3:
                        continue
                    label = cells[0].get_text(" ", strip=True)
                    if label == "Input price (text, image, video)":
                        label = "Input price"
                    key = {
                        "Input price": "input",
                        "Output price (including thinking tokens)": "output",
                        "Context caching price": "cache",
                    }.get(label)
                    if key:
                        prices[key] = _rates(cells[2])
                if {"input", "output"} <= prices.keys():
                    tiers[tier] = prices
            except ValueError:
                continue
        if tiers:
            models[model] = {"tiers": tiers}
    if not models:
        raise ValueError("Official page has no recognizable text model prices")
    return {"currency": "USD", "models": models, "source": SOURCE}


def select_rate(
    records: list[dict[str, Any]], *, input_tokens: int, day: str
) -> float | None:
    matches = [
        r["usd_per_1m"]
        for r in records
        if r.get("valid_from", "0000") <= day <= r.get("valid_until", "9999")
        and input_tokens <= r.get("max_input", float("inf"))
        and input_tokens > r.get("min_input_exclusive", -1)
    ]
    return float(matches[0]) if len(matches) == 1 else None


async def refresh_prices() -> PricingStatus:
    global _snapshot, _error
    if os.getenv("EVALUATION_PRICE_SNAPSHOT_PATH"):
        return pricing_status()
    async with _lock:
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                response = await client.get(SOURCE, params={"hl": "en"})
                response.raise_for_status()
            snapshot = await asyncio.to_thread(parse_prices, response.text)
            snapshot["fetched_at"] = datetime.now(timezone.utc).isoformat()
            snapshot["snapshot_id"] = (
                "google-"
                + hashlib.sha256(
                    json.dumps(snapshot, sort_keys=True).encode()
                ).hexdigest()[:20]
            )
            await asyncio.to_thread(_save, snapshot)
            _snapshot, _error = snapshot, None
        except (httpx.HTTPError, ValueError, OSError):
            _error = "sync_failed"
            logger.warning(
                "Gemini price refresh failed; retaining last price snapshot",
                exc_info=True,
            )
    return pricing_status()


def _save(snapshot: dict[str, Any]) -> None:
    directory = _directory()
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(snapshot, ensure_ascii=False, indent=2)
    (directory / f"{snapshot['snapshot_id']}.json").write_text(
        payload, encoding="utf-8"
    )
    temporary = directory / f"{uuid4().hex}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(directory / "current.json")


async def refresh_loop() -> None:
    while True:
        if not os.getenv("EVALUATION_PRICE_SNAPSHOT_PATH"):
            await refresh_prices()
        await asyncio.sleep(86400)
