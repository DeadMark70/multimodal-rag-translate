"""Utilities to upgrade the RAGAS master dataset and derive a deterministic ready dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MASTER_DATASET_VERSION = "2.0.0"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _clean_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _short_index(short_payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for item in short_payload:
        question_id = item.get("id")
        if isinstance(question_id, str) and question_id.strip():
            index[question_id] = item
    return index


def upgrade_master_dataset(
    master_payload: dict[str, Any],
    short_payload: list[dict[str, Any]],
) -> dict[str, Any]:
    upgraded = json.loads(json.dumps(master_payload, ensure_ascii=False))
    metadata = upgraded.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        upgraded["metadata"] = metadata
    metadata["dataset_version"] = MASTER_DATASET_VERSION
    metadata["dataset_role"] = "master"

    questions = upgraded.get("questions")
    if not isinstance(questions, list):
        raise ValueError("master dataset must contain a questions array")

    short_by_id = _short_index(short_payload)
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("master questions entries must be objects")
        question_id = question.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError("every master question must contain a stable id")
        short_entry = short_by_id.get(question_id, {})
        question["ground_truth_short"] = _clean_optional_text(short_entry.get("ground_truth_short"))
        question["key_points"] = _clean_list(short_entry.get("key_points"))
        question["ragas_focus"] = _clean_list(short_entry.get("ragas_focus"))

    metadata["total_questions"] = len(questions)
    return upgraded


def build_ragas_ready_dataset(master_payload: dict[str, Any], *, derived_from: str) -> dict[str, Any]:
    metadata = master_payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    questions = master_payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("master dataset must contain a questions array")

    derived_at = (
        metadata.get("created")
        or metadata.get("updated")
        or metadata.get("exported_at")
        or metadata.get("dataset_version")
        or MASTER_DATASET_VERSION
    )

    ready_questions: list[dict[str, Any]] = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("master questions entries must be objects")
        ready_questions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "ground_truth": question.get("ground_truth_short") or question.get("ground_truth") or "",
                "key_points": _clean_list(question.get("key_points")),
                "category": question.get("category"),
                "ragas_focus": _clean_list(question.get("ragas_focus")),
            }
        )

    return {
        "metadata": {
            "dataset_version": metadata.get("dataset_version", MASTER_DATASET_VERSION),
            "dataset_role": "ragas_ready",
            "derived_from": derived_from,
            "derived_at": derived_at,
            "total_questions": len(ready_questions),
        },
        "questions": ready_questions,
    }


def generate_ragas_datasets(
    *,
    master_path: Path,
    short_path: Path,
    ready_path: Path,
    write_master: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    master_payload = _load_json(master_path)
    short_payload = _load_json(short_path)
    if not isinstance(short_payload, list):
        raise ValueError("short dataset must be a list of question supplements")

    upgraded_master = upgrade_master_dataset(master_payload, short_payload)
    ready_payload = build_ragas_ready_dataset(upgraded_master, derived_from=master_path.name)

    if write_master:
        _write_json(master_path, upgraded_master)
    _write_json(ready_path, ready_payload)
    return upgraded_master, ready_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade the master RAGAS dataset and derive ragas_ready.json")
    parser.add_argument("--master", type=Path, required=True, help="Path to ragasfullqa.json")
    parser.add_argument("--short", type=Path, required=True, help="Path to ragasshortqa.json")
    parser.add_argument("--ready", type=Path, required=True, help="Output path for ragas_ready.json")
    parser.add_argument(
        "--skip-master-write",
        action="store_true",
        help="Do not overwrite the master dataset after merging short fields",
    )
    args = parser.parse_args()

    generate_ragas_datasets(
        master_path=args.master,
        short_path=args.short,
        ready_path=args.ready,
        write_master=not args.skip_master_write,
    )


if __name__ == "__main__":
    main()
