from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from core import google_genai_client
from evaluation import model_discovery


class _FlakyPager:
    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)

    def __iter__(self) -> "_FlakyPager":
        return self

    def __next__(self) -> object:
        item = next(self._items)
        if isinstance(item, Exception):
            raise item
        return item


def _raw_model(
    name: str,
    *,
    methods: list[str] | None = None,
    display_name: str | None = None,
    description: str | None = None,
    input_token_limit: int | None = None,
    output_token_limit: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        display_name=display_name,
        description=description,
        input_token_limit=input_token_limit,
        output_token_limit=output_token_limit,
        supported_generation_methods=methods or ["generateContent"],
    )


def _reset_state() -> None:
    model_discovery._cache = None
    google_genai_client.reset_google_genai_client_cache()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_state()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    _reset_state()


def test_get_google_genai_client_reuses_cached_client_and_rebuilds_on_key_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key-one")
    first_client = Mock(name="first-client")
    second_client = Mock(name="second-client")

    with patch(
        "core.google_genai_client.genai.Client",
        side_effect=[first_client, second_client],
    ) as mock_client:
        first = google_genai_client.get_google_genai_client()
        second = google_genai_client.get_google_genai_client()

        monkeypatch.setenv("GOOGLE_API_KEY", "key-two")
        third = google_genai_client.get_google_genai_client()

    assert first is first_client
    assert second is first_client
    assert third is second_client
    assert mock_client.call_count == 2
    assert mock_client.call_args_list[0].kwargs["api_key"] == "key-one"
    assert mock_client.call_args_list[1].kwargs["api_key"] == "key-two"


def test_get_google_genai_client_uses_gemini_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    client = Mock(name="gemini-client")

    with patch("core.google_genai_client.genai.Client", return_value=client) as mock_client:
        actual = google_genai_client.get_google_genai_client()

    assert actual is client
    assert mock_client.call_args.kwargs["api_key"] == "gemini-key"


def test_fetch_models_sync_skips_unparseable_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key-one")
    good_one = _raw_model(
        "models/gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
    )
    good_two = _raw_model(
        "models/gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash Lite",
    )
    pager = _FlakyPager([good_one, ValueError("bad entry"), good_two])
    client = Mock(name="client")
    client.models.list.return_value = pager

    with patch("evaluation.model_discovery.get_google_genai_client", return_value=client):
        actual = model_discovery._fetch_models_sync()

    assert actual == [good_one, good_two]
    client.models.list.assert_called_once_with(config={"page_size": 100})


@pytest.mark.asyncio
async def test_list_available_models_filters_and_normalizes_sdk_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key-one")
    client = Mock(name="client")
    client.models.list.return_value = [
        _raw_model(
            "models/gemini-2.5-flash-lite",
            display_name="Gemini 2.5 Flash Lite",
            description="Lite model",
            input_token_limit=1_048_576,
            output_token_limit=8_192,
        ),
        _raw_model(
            "models/embedding-only",
            methods=["embedContent"],
            display_name="Embedding Only",
        ),
        _raw_model(
            "models/gemini-2.5-flash",
            display_name="Gemini 2.5 Flash",
            description="Fast model",
            input_token_limit=1_048_576,
            output_token_limit=8_192,
        ),
    ]

    with patch("evaluation.model_discovery.get_google_genai_client", return_value=client):
        actual = await model_discovery.list_available_models()

    assert [item.name for item in actual] == [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    assert actual[0].supported_actions == ["generateContent"]
    assert actual[0].display_name == "Gemini 2.5 Flash"


@pytest.mark.asyncio
async def test_list_available_models_uses_fallback_when_no_api_key() -> None:
    with patch("evaluation.model_discovery.get_google_genai_client", return_value=None) as mock_client:
        actual = await model_discovery.list_available_models()

    assert [item.name for item in actual] == [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]
    mock_client.assert_called_once_with()


@pytest.mark.asyncio
async def test_list_available_models_uses_fallback_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key-one")

    with patch(
        "evaluation.model_discovery._fetch_models_sync",
        side_effect=RuntimeError("boom"),
    ):
        actual = await model_discovery.list_available_models()

    assert [item.name for item in actual] == [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
    ]


@pytest.mark.asyncio
async def test_list_available_models_uses_cache_until_force_refresh() -> None:
    raw_model = _raw_model(
        "models/gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
    )

    with patch(
        "evaluation.model_discovery._fetch_models_sync",
        return_value=[raw_model],
    ) as mock_fetch:
        first = await model_discovery.list_available_models()
        second = await model_discovery.list_available_models()
        third = await model_discovery.list_available_models(force_refresh=True)

    assert mock_fetch.call_count == 2
    assert [item.name for item in first] == ["gemini-2.5-flash"]
    assert [item.name for item in second] == ["gemini-2.5-flash"]
    assert [item.name for item in third] == ["gemini-2.5-flash"]
    assert first is not second
    assert second is not third



@pytest.mark.asyncio
async def test_list_available_models_reuses_last_successful_cache_when_refresh_fails() -> None:
    raw_model = _raw_model(
        "models/gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
    )

    with patch(
        "evaluation.model_discovery._fetch_models_sync",
        return_value=[raw_model],
    ):
        first = await model_discovery.list_available_models(force_refresh=True)

    with patch(
        "evaluation.model_discovery._fetch_models_sync",
        side_effect=RuntimeError("boom"),
    ):
        second = await model_discovery.list_available_models(force_refresh=True)

    assert [item.name for item in first] == ["gemini-2.5-pro"]
    assert [item.name for item in second] == ["gemini-2.5-pro"]


