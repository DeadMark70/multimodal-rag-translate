"""Application-level error types and handlers."""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi.exceptions import RequestValidationError
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class ErrorCode(str, Enum):
    """Canonical API error codes."""

    BAD_REQUEST = "BAD_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    DATABASE_ERROR = "DATABASE_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    AUTH_SERVICE_UNAVAILABLE = "AUTH_SERVICE_UNAVAILABLE"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class AppError(Exception):
    """Application error with explicit status and code."""

    def __init__(
        self,
        *,
        code: ErrorCode,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details


def build_error_response(
    *,
    request: Request,
    code: ErrorCode | str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Builds a standardized error response envelope."""
    request_id = getattr(request.state, "request_id", None)
    code_value = code.value if isinstance(code, ErrorCode) else str(code)
    payload: dict[str, Any] = {
        "error": {
            "code": code_value,
            "message": message,
            "request_id": request_id,
        }
    }
    if details:
        payload["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=payload)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """FastAPI exception handler for AppError."""
    return build_error_response(
        request=request,
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        details=exc.details,
    )


def _map_status_to_code(status_code: int) -> ErrorCode:
    """Maps HTTP status to canonical error code."""
    if status_code == 400:
        return ErrorCode.BAD_REQUEST
    if status_code == 401:
        return ErrorCode.UNAUTHORIZED
    if status_code == 403:
        return ErrorCode.FORBIDDEN
    if status_code == 404:
        return ErrorCode.NOT_FOUND
    if status_code == 422:
        return ErrorCode.VALIDATION_ERROR
    if 400 <= status_code < 500:
        return ErrorCode.BAD_REQUEST
    return ErrorCode.INTERNAL_ERROR


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Converts HTTPException responses into the standard error envelope."""
    details: dict[str, Any] | None = None
    if isinstance(exc.detail, str):
        message = exc.detail
    elif isinstance(exc.detail, dict):
        message = str(exc.detail.get("message") or "Request failed")
        details = exc.detail
    else:
        message = "Request failed"
        if exc.detail is not None:
            details = {"detail": exc.detail}

    return build_error_response(
        request=request,
        code=_map_status_to_code(exc.status_code),
        message=message,
        status_code=exc.status_code,
        details=details,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Converts request validation errors into the standard error envelope."""
    return build_error_response(
        request=request,
        code=ErrorCode.VALIDATION_ERROR,
        message="Request validation failed",
        status_code=422,
        details={"errors": exc.errors()},
    )


async def unhandled_exception_handler(
    request: Request, _: Exception
) -> JSONResponse:
    """Fallback error handler for unexpected exceptions."""
    return build_error_response(
        request=request,
        code=ErrorCode.INTERNAL_ERROR,
        message="Internal server error",
        status_code=500,
    )
