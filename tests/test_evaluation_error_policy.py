import httpx
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel, ValidationError

from evaluation.error_policy import classify_evaluation_error, retry_delay_seconds


def test_rate_limit_is_retryable() -> None:
    decision = classify_evaluation_error(google_exceptions.ResourceExhausted("quota"))
    assert decision.error_type == "rate_limit"
    assert decision.retryable is True


def test_transport_error_is_retryable() -> None:
    decision = classify_evaluation_error(httpx.ConnectError("offline"))
    assert decision.error_type == "transport"
    assert decision.retryable is True


def test_authentication_error_is_permanent() -> None:
    exc = type("ProviderAuthError", (Exception,), {"status_code": 401})("bad key")
    decision = classify_evaluation_error(exc)
    assert decision.error_type == "authentication"
    assert decision.retryable is False


def test_retry_after_takes_precedence() -> None:
    assert retry_delay_seconds(3, 17.0) == 17.0


def test_provider_validation_error_is_not_reclassified_as_dataset_input() -> None:
    class ProviderRequest(BaseModel):
        thinking_config: int

    try:
        ProviderRequest.model_validate({"thinking_config": "minimal"})
    except ValidationError as exc:
        decision = classify_evaluation_error(exc)
    else:  # pragma: no cover - the invalid fixture must always reject.
        raise AssertionError("expected provider request validation to fail")

    assert decision.error_type == "unknown"
    assert decision.safe_message == "An unexpected evaluation error occurred."
