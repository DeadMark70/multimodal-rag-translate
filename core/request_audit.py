"""Safe request-context and audit logging helpers."""

import hashlib
import ipaddress
import json
import logging
import re
import time
import uuid

from fastapi import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def normalize_request_id(value: str | None) -> str:
    if value and _SAFE_REQUEST_ID.fullmatch(value):
        return value
    return str(uuid.uuid4())


def anonymize_user_id(value: str | None) -> str:
    if not value:
        return "anonymous"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _client_ip(request: Request) -> str | None:
    forwarded_ip = request.headers.get("X-Real-IP")
    if forwarded_ip:
        try:
            return str(ipaddress.ip_address(forwarded_ip))
        except ValueError:
            pass
    return request.client.host if request.client else None


async def request_context_middleware(request: Request, call_next) -> Response:
    request_id = normalize_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    started_at = time.perf_counter()
    response: Response | None = None
    status = 500

    try:
        response = await call_next(request)
        status = response.status_code
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", None) or "<unmatched>"
        audit_payload = {
            "event": "request_complete",
            "request_id": request_id,
            "method": request.method,
            "path": path,
            "status": status,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "client_ip": _client_ip(request),
            "user_id": anonymize_user_id(
                getattr(request.state, "audit_user_id", None)
            ),
        }
        logger.info(json.dumps(audit_payload, separators=(",", ":")))

    if response is not None:
        response.headers["X-Request-ID"] = request_id
        return response
    raise RuntimeError("Request middleware completed without a response")
