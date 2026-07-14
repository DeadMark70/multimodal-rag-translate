import httpx
from google.api_core import exceptions as google_exceptions

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
