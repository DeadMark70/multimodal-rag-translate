"""Local JSON storage for evaluation test cases and model configs."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.errors import AppError, ErrorCode

BASE_UPLOAD_FOLDER = "uploads"
_VALID_USER_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_GUARD = asyncio.Lock()


def _default_dataset() -> dict[str, Any]:
    now = datetime.now(timezone.utc).date().isoformat()
    return {
        "metadata": {
            "created": now,
            "version": "1.0",
            "total_questions": 0,
        },
        "questions": [],
    }


def _default_model_configs() -> dict[str, Any]:
    return {"items": []}


def _normalize_user_id(user_id: str) -> str:
    if not _VALID_USER_ID.fullmatch(user_id):
        raise AppError(
            code=ErrorCode.BAD_REQUEST,
            message="Invalid user id format",
            status_code=400,
        )
    return user_id


def _evaluation_dir(user_id: str) -> Path:
    safe_user_id = _normalize_user_id(user_id)
    return Path(BASE_UPLOAD_FOLDER) / safe_user_id / "evaluation"


def _test_case_path(user_id: str) -> Path:
    return _evaluation_dir(user_id) / "test_cases.json"


def _model_config_path(user_id: str) -> Path:
    return _evaluation_dir(user_id) / "model_configs.json"


def _read_json(path: Path, default_value: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return copy.deepcopy(default_value)
    try:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError as exc:
        raise AppError(
            code=ErrorCode.PROCESSING_ERROR,
            message=f"Corrupted JSON file: {path.name}",
            status_code=500,
        ) from exc

    raise AppError(
        code=ErrorCode.PROCESSING_ERROR,
        message=f"Invalid JSON content: {path.name}",
        status_code=500,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


async def _get_user_lock(user_id: str) -> asyncio.Lock:
    async with _LOCKS_GUARD:
        return _LOCKS.setdefault(user_id, asyncio.Lock())


def _normalize_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    metadata = dataset.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    questions = dataset.get("questions")
    if not isinstance(questions, list):
        questions = []
    metadata["total_questions"] = len(questions)
    return {"metadata": metadata, "questions": questions}


async def list_test_cases(user_id: str) -> list[dict[str, Any]]:
    lock = await _get_user_lock(user_id)
    async with lock:
        dataset = await asyncio.to_thread(_read_json, _test_case_path(user_id), _default_dataset())
        normalized = _normalize_dataset(dataset)
        return normalized["questions"]


async def create_test_case(user_id: str, test_case: dict[str, Any]) -> dict[str, Any]:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _test_case_path(user_id)
        dataset = await asyncio.to_thread(_read_json, path, _default_dataset())
        normalized = _normalize_dataset(dataset)
        questions = normalized["questions"]

        candidate = dict(test_case)
        candidate["id"] = candidate.get("id") or str(uuid4())
        if any(item.get("id") == candidate["id"] for item in questions):
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Test case id already exists: {candidate['id']}",
                status_code=400,
            )

        questions.append(candidate)
        normalized["metadata"]["total_questions"] = len(questions)
        await asyncio.to_thread(_atomic_write_json, path, normalized)
        return candidate


async def import_test_cases(
    user_id: str,
    questions_to_import: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> tuple[int, int]:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _test_case_path(user_id)
        dataset = await asyncio.to_thread(_read_json, path, _default_dataset())
        normalized = _normalize_dataset(dataset)

        existing = {item["id"]: item for item in normalized["questions"] if isinstance(item, dict) and item.get("id")}
        imported_count = 0

        for item in questions_to_import:
            candidate = dict(item)
            candidate["id"] = candidate.get("id") or str(uuid4())
            existing[candidate["id"]] = candidate
            imported_count += 1

        merged_questions = list(existing.values())
        merged_metadata = dict(normalized["metadata"])
        if metadata:
            merged_metadata.update(metadata)
        merged_metadata["total_questions"] = len(merged_questions)

        updated = {"metadata": merged_metadata, "questions": merged_questions}
        await asyncio.to_thread(_atomic_write_json, path, updated)
        return imported_count, len(merged_questions)


async def update_test_case(
    user_id: str,
    test_case_id: str,
    test_case: dict[str, Any],
) -> dict[str, Any]:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _test_case_path(user_id)
        dataset = await asyncio.to_thread(_read_json, path, _default_dataset())
        normalized = _normalize_dataset(dataset)

        for index, item in enumerate(normalized["questions"]):
            if item.get("id") == test_case_id:
                candidate = dict(test_case)
                candidate["id"] = test_case_id
                normalized["questions"][index] = candidate
                normalized["metadata"]["total_questions"] = len(normalized["questions"])
                await asyncio.to_thread(_atomic_write_json, path, normalized)
                return candidate

        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Test case not found",
            status_code=404,
        )


async def delete_test_case(user_id: str, test_case_id: str) -> int:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _test_case_path(user_id)
        dataset = await asyncio.to_thread(_read_json, path, _default_dataset())
        normalized = _normalize_dataset(dataset)
        original_len = len(normalized["questions"])
        normalized["questions"] = [
            item for item in normalized["questions"] if item.get("id") != test_case_id
        ]
        if len(normalized["questions"]) == original_len:
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Test case not found",
                status_code=404,
            )

        normalized["metadata"]["total_questions"] = len(normalized["questions"])
        await asyncio.to_thread(_atomic_write_json, path, normalized)
        return len(normalized["questions"])


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def list_model_configs(user_id: str) -> list[dict[str, Any]]:
    lock = await _get_user_lock(user_id)
    async with lock:
        store = await asyncio.to_thread(_read_json, _model_config_path(user_id), _default_model_configs())
        items = store.get("items")
        if not isinstance(items, list):
            return []
        return items


async def create_model_config(user_id: str, model_config: dict[str, Any]) -> dict[str, Any]:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _model_config_path(user_id)
        store = await asyncio.to_thread(_read_json, path, _default_model_configs())
        items = store.get("items")
        if not isinstance(items, list):
            items = []

        now = _utc_now_iso()
        candidate = dict(model_config)
        candidate["id"] = candidate.get("id") or str(uuid4())
        candidate["created_at"] = candidate.get("created_at") or now
        candidate["updated_at"] = now

        if any(item.get("id") == candidate["id"] for item in items):
            raise AppError(
                code=ErrorCode.BAD_REQUEST,
                message=f"Model config id already exists: {candidate['id']}",
                status_code=400,
            )

        items.append(candidate)
        await asyncio.to_thread(_atomic_write_json, path, {"items": items})
        return candidate


async def update_model_config(
    user_id: str,
    config_id: str,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _model_config_path(user_id)
        store = await asyncio.to_thread(_read_json, path, _default_model_configs())
        items = store.get("items")
        if not isinstance(items, list):
            items = []

        for index, item in enumerate(items):
            if item.get("id") == config_id:
                candidate = dict(model_config)
                candidate["id"] = config_id
                candidate["created_at"] = item.get("created_at") or _utc_now_iso()
                candidate["updated_at"] = _utc_now_iso()
                items[index] = candidate
                await asyncio.to_thread(_atomic_write_json, path, {"items": items})
                return candidate

        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="Model config not found",
            status_code=404,
        )


async def delete_model_config(user_id: str, config_id: str) -> int:
    lock = await _get_user_lock(user_id)
    async with lock:
        path = _model_config_path(user_id)
        store = await asyncio.to_thread(_read_json, path, _default_model_configs())
        items = store.get("items")
        if not isinstance(items, list):
            items = []

        filtered = [item for item in items if item.get("id") != config_id]
        if len(filtered) == len(items):
            raise AppError(
                code=ErrorCode.NOT_FOUND,
                message="Model config not found",
                status_code=404,
            )

        await asyncio.to_thread(_atomic_write_json, path, {"items": filtered})
        return len(filtered)

