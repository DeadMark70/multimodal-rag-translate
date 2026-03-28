from __future__ import annotations

import json
from pathlib import Path

from evaluation.dataset_generator import generate_ragas_datasets


def test_generate_ragas_datasets_merges_short_fields_and_is_idempotent(tmp_path: Path) -> None:
    master_path = tmp_path / "ragasfullqa.json"
    short_path = tmp_path / "ragasshortqa.json"
    ready_path = tmp_path / "ragas_ready.json"

    master_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "created": "2026-03-28",
                    "version": "1.0",
                    "total_questions": 1,
                },
                "questions": [
                    {
                        "id": "Q1",
                        "question": "Question 1",
                        "ground_truth": "Long answer",
                        "category": "綜合比較題",
                        "source_docs": ["doc-a.pdf"],
                        "difficulty": "hard",
                        "requires_multi_doc_reasoning": True,
                        "test_objective": "objective",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    short_path.write_text(
        json.dumps(
            [
                {
                    "id": "Q1",
                    "ground_truth_short": "Short answer",
                    "key_points": ["point-1", "point-2"],
                    "ragas_focus": ["answer_correctness", "faithfulness"],
                }
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    upgraded_master, ready_payload = generate_ragas_datasets(
        master_path=master_path,
        short_path=short_path,
        ready_path=ready_path,
    )

    assert upgraded_master["metadata"]["dataset_version"] == "2.0.0"
    assert upgraded_master["metadata"]["dataset_role"] == "master"
    assert upgraded_master["questions"][0]["ground_truth_short"] == "Short answer"
    assert upgraded_master["questions"][0]["key_points"] == ["point-1", "point-2"]
    assert upgraded_master["questions"][0]["ragas_focus"] == ["answer_correctness", "faithfulness"]
    assert ready_payload["metadata"] == {
        "dataset_version": "2.0.0",
        "dataset_role": "ragas_ready",
        "derived_from": "ragasfullqa.json",
        "derived_at": "2026-03-28",
        "total_questions": 1,
    }
    assert ready_payload["questions"][0]["ground_truth"] == "Short answer"

    first_ready = ready_path.read_text(encoding="utf-8")
    generate_ragas_datasets(master_path=master_path, short_path=short_path, ready_path=ready_path)
    second_ready = ready_path.read_text(encoding="utf-8")
    assert first_ready == second_ready
