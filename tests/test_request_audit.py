import json
import logging
import re

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from core.auth import get_current_user_id
from core.errors import AppError, ErrorCode
from core.request_audit import (
    anonymize_user_id,
    normalize_request_id,
    request_context_middleware,
)


def test_request_id_accepts_safe_bounded_value() -> None:
    assert normalize_request_id("req-123_OK.test") == "req-123_OK.test"


def test_request_id_replaces_unsafe_or_oversized_value() -> None:
    generated = normalize_request_id("bad\nvalue")

    assert generated != "bad\nvalue"
    assert re.fullmatch(r"[0-9a-f-]{36}", generated)
    assert normalize_request_id("x" * 65) != "x" * 65


def test_user_id_is_anonymous_or_stably_hashed() -> None:
    assert anonymize_user_id(None) == "anonymous"
    assert anonymize_user_id("user-123") == anonymize_user_id("user-123")
    assert "user-123" not in anonymize_user_id("user-123")
    assert len(anonymize_user_id("user-123")) == 12


def test_request_audit_logs_safe_route_metadata_without_request_secrets(caplog) -> None:
    app = FastAPI()
    app.middleware("http")(request_context_middleware)

    @app.get("/items/{item_id}")
    async def get_item(item_id: int, request: Request) -> dict[str, int]:
        request.state.audit_user_id = "user-secret"
        return {"item_id": item_id}

    with caplog.at_level(logging.INFO, logger="core.request_audit"):
        response = TestClient(app).get(
            "/items/42?question=secret-prompt",
            headers={
                "Authorization": "Bearer secret-token",
                "X-Real-IP": "203.0.113.9",
            },
        )

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "core.request_audit"
    ]

    assert response.status_code == 200
    assert len(records) == 1
    assert records[0]["event"] == "request_complete"
    assert records[0]["method"] == "GET"
    assert records[0]["path"] == "/items/{item_id}"
    assert records[0]["status"] == 200
    assert records[0]["client_ip"] == "203.0.113.9"
    assert records[0]["request_id"] == response.headers["X-Request-ID"]
    assert "secret-prompt" not in caplog.text
    assert "secret-token" not in caplog.text
    assert "user-secret" not in caplog.text


def test_request_audit_redacts_unmatched_path_segments(caplog) -> None:
    app = FastAPI()
    app.middleware("http")(request_context_middleware)
    sensitive_path = "/uploads/private-report.pdf/token-sk-live-secret"

    with caplog.at_level(logging.INFO, logger="core.request_audit"):
        response = TestClient(app).get(sensitive_path)

    records = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "core.request_audit"
    ]

    assert response.status_code == 404
    assert len(records) == 1
    assert records[0]["path"] == "<unmatched>"
    assert "private-report.pdf" not in caplog.text
    assert "token-sk-live-secret" not in caplog.text


def test_successful_authentication_attaches_user_id_to_request_state(monkeypatch) -> None:
    app = FastAPI()

    async def fetch_user_id_from_token(token: str) -> str:
        assert token == "secret-token"
        return "user-123"

    monkeypatch.setattr("core.auth.fetch_user_id_from_token", fetch_user_id_from_token)

    @app.get("/authenticated")
    async def authenticated(
        request: Request,
        user_id: str = Depends(get_current_user_id),
    ) -> dict[str, str]:
        return {"user_id": user_id, "audit_user_id": request.state.audit_user_id}

    response = TestClient(app).get(
        "/authenticated",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert response.json() == {"user_id": "user-123", "audit_user_id": "user-123"}


def test_invalid_authentication_does_not_log_raw_token(monkeypatch, caplog) -> None:
    app = FastAPI()
    app.middleware("http")(request_context_middleware)

    async def reject_token(token: str) -> str:
        raise AppError(
            code=ErrorCode.UNAUTHORIZED,
            message="Authentication failed",
            status_code=401,
        )

    monkeypatch.setattr("core.auth.fetch_user_id_from_token", reject_token)

    @app.get("/authenticated")
    async def authenticated(user_id: str = Depends(get_current_user_id)) -> dict[str, str]:
        return {"user_id": user_id}

    with caplog.at_level(logging.INFO, logger="core.request_audit"):
        response = TestClient(app, raise_server_exceptions=False).get(
            "/authenticated",
            headers={"Authorization": "Bearer raw-invalid-token"},
        )

    assert response.status_code == 500
    assert "raw-invalid-token" not in caplog.text
