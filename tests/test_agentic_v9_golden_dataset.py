"""Integrity checks for the immutable Agentic v9 evaluation inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "evaluation" / "golden"
QUESTIONS_PATH = GOLDEN_DIR / "agentic_v9_questions_v2.json"
ROUTES_PATH = GOLDEN_DIR / "agentic_v9_route_regressions.json"
MANIFEST_PATH = GOLDEN_DIR / "agentic_v9_baseline_manifest.json"
FORMAL_ROUTES = {
    "single_lookup",
    "bounded_compare",
    "exact_structured",
    "multi_document_exact",
    "multi_hop",
    "graph_relational",
}
EXPECTED_QUESTIONS_SHA256 = "3c7f157f06db24de6d1bc209aaf91809fde1e542c1292c12a9ebb126b4d66a21"
EXPECTED_ROUTES_SHA256 = "1cafdbd14c0391a400354a58672829ea84d41aead40feca399a3a7a1aa4810c3"
EXPECTED_MANIFEST_SHA256 = "ba31ec465f3199116c14b55251cd60e30e41de52f55ac863e6d2781d13067dfd"
CANONICAL_BASELINE_MANIFEST = {
    "schema_version": "1",
    "baseline_kind": "agentic-v9-evidence-first-preimplementation",
    "backend": {
        "behavioral_commit": "651acc0",
        "branch": "feature/agentic-v9-evidence-first",
    },
    "frontend": {
        "url": "https://github.com/DeadMark70/Multimodal_RAG_System_Web.git",
        "branch": "master",
        "commit": "1ab15449af756886039614fab6b6cc64781d1d23",
    },
    "model": {
        "config_id": "a4cd737f-790e-43b5-a897-a2adb25a8514",
        "name": "testing2",
        "model_name": "gemini-2.5-flash-lite",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "max_input_tokens": 1048576,
        "max_output_tokens": 8192,
        "thinking_mode": False,
        "thinking_budget": 8192,
        "thinking_level": None,
        "thinking_include_thoughts": False,
    },
    "execution_profiles": {
        "naive": {
            "execution_profile": None,
            "context_policy_version": "v3_answer_aware_pack",
        },
        "advanced": {
            "execution_profile": "advanced_eval_v2_multiquery_recursive_baseline",
            "context_policy_version": "v3_answer_aware_pack",
        },
        "graph": {
            "execution_profile": "graph_eval_v2_multiquery_locator_recursive_baseline",
            "context_policy_version": "v3_answer_aware_pack",
        },
        "agentic": {
            "execution_profile": "agentic_eval_v8_multiquery_locator_recursive_baseline",
            "context_policy_version": "v4_semantic_router_gate",
        },
    },
    "corpus_and_index": {
        "knowledge_base_id": None,
        "index_version": None,
        "retriever_config_hash": None,
        "capture_status": "unavailable_in_source_campaign_rows",
    },
    "prompt_snapshot": {
        "prompt_pack_version": None,
        "capture_status": "unavailable_in_source_campaign_rows",
    },
    "evaluator_snapshot": {
        "test_suite_id": None,
        "ragas_config": None,
        "capture_status": "unavailable_in_source_campaign_rows",
    },
    "source_campaign_ids": ["ffd6c442-663e-4041-83be-d352d937c839"],
    "artifacts": {
        "agentic_v9_questions_v2.json": EXPECTED_QUESTIONS_SHA256,
        "agentic_v9_route_regressions.json": EXPECTED_ROUTES_SHA256,
    },
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_v9_golden_paths_and_hashes_are_frozen() -> None:
    for path in (QUESTIONS_PATH, ROUTES_PATH, MANIFEST_PATH):
        assert path.is_file(), f"missing frozen evaluation artifact: {path.name}"

    manifest = _load(MANIFEST_PATH)
    assert _sha256(QUESTIONS_PATH) == EXPECTED_QUESTIONS_SHA256
    assert _sha256(ROUTES_PATH) == EXPECTED_ROUTES_SHA256
    assert _sha256(MANIFEST_PATH) == EXPECTED_MANIFEST_SHA256
    assert manifest == CANONICAL_BASELINE_MANIFEST


def test_agentic_v9_formal_questions_are_complete_and_referentially_valid() -> None:
    payload = _load(QUESTIONS_PATH)
    questions = payload["questions"]
    assert payload["metadata"]["dataset_version"] == "2.0.0"
    assert len(questions) == 16
    assert {question["id"] for question in questions} == {f"Q{number}" for number in range(1, 17)}

    for question in questions:
        assert question["source_docs"]
        assert question["expected_route"] in FORMAL_ROUTES - {"graph_relational"}
        facts = question["atomic_facts"]
        evidence = question["expected_evidence"]
        assert facts and evidence
        fact_ids = {fact["id"] for fact in facts}
        assert len(fact_ids) == len(facts)
        assert any(fact["required"] for fact in facts)
        for fact in facts:
            assert fact["text"]
            assert fact["claim_importance"] in {"critical", "high", "medium"}
        for item in evidence:
            assert item["id"]
            assert item["doc_id"] in question["source_docs"]
            assert item["locator"]
            assert set(item["supports_fact_ids"]).issubset(fact_ids)

        resolutions = question.get("expected_slot_resolutions", [])
        unavailable_fact_ids = {
            resolution["slot_id"]
            for resolution in resolutions
            if resolution["status"] in {"explicitly_unavailable", "not_found"}
        }
        assert all(
            not unavailable_fact_ids.intersection(item["supports_fact_ids"])
            for item in evidence
        )
        for resolution in resolutions:
            assert resolution["slot_id"] in fact_ids
            assert resolution["status"] in {"explicitly_unavailable", "not_found"}
            assert resolution["reason"]

    q14 = next(question for question in questions if question["id"] == "Q14")
    assert "SegmentAnyBone" not in q14["source_docs"]
    assert any("insufficient" in fact["text"].lower() for fact in q14["atomic_facts"])
    assert any("original SAM" in fact["text"] for fact in q14["atomic_facts"])
    assert {resolution["slot_id"] for resolution in q14["expected_slot_resolutions"]} == {
        "Q14-F2",
        "Q14-F3",
    }
    assert all(
        not set(item["supports_fact_ids"]).intersection({"Q14-F2", "Q14-F3"})
        for item in q14["expected_evidence"]
    )

    q16 = next(question for question in questions if question["id"] == "Q16")
    formula = next(fact["text"] for fact in q16["atomic_facts"] if fact["id"] == "Q16-F2")
    assert "P(x,y)" in formula and "A^c(x,y)" in formula
    assert "\n" not in formula


def test_agentic_v9_route_regressions_cover_all_routes_without_polluting_formal_scores() -> None:
    payload = _load(ROUTES_PATH)
    cases = payload["cases"]
    case_ids = [case["id"] for case in cases]
    assert len(case_ids) == len(set(case_ids))
    assert len(cases) == len(FORMAL_ROUTES)
    assert {case["expected_route"] for case in cases} == FORMAL_ROUTES
    synthetic_cases = [case for case in cases if case["synthetic"]]
    assert len(synthetic_cases) == 1
    assert synthetic_cases[0]["expected_route"] == "graph_relational"
    assert all(not case["include_in_formal_metrics"] for case in cases)
    assert all(
        case["requires_graph"] == (case["expected_route"] == "graph_relational")
        for case in cases
    )
