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


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_agentic_v9_golden_paths_and_hashes_are_frozen() -> None:
    for path in (QUESTIONS_PATH, ROUTES_PATH, MANIFEST_PATH):
        assert path.is_file(), f"missing frozen evaluation artifact: {path.name}"

    manifest = _load(MANIFEST_PATH)
    assert manifest["schema_version"] == "1"
    assert manifest["frontend"]["url"] == "https://github.com/DeadMark70/Multimodal_RAG_System_Web.git"
    assert manifest["frontend"]["branch"] == "master"
    assert manifest["frontend"]["commit"] == "1ab15449af756886039614fab6b6cc64781d1d23"
    assert manifest["source_campaign_ids"] == ["ffd6c442-663e-4041-83be-d352d937c839"]
    assert manifest["artifacts"] == {
        QUESTIONS_PATH.name: _sha256(QUESTIONS_PATH),
        ROUTES_PATH.name: _sha256(ROUTES_PATH),
    }


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

    q14 = next(question for question in questions if question["id"] == "Q14")
    assert "SegmentAnyBone" not in q14["source_docs"]
    assert any("insufficient" in fact["text"].lower() for fact in q14["atomic_facts"])
    assert any("original SAM" in fact["text"] for fact in q14["atomic_facts"])

    q16 = next(question for question in questions if question["id"] == "Q16")
    formula = next(fact["text"] for fact in q16["atomic_facts"] if fact["id"] == "Q16-F2")
    assert "P(x,y)" in formula and "A^c(x,y)" in formula
    assert "\n" not in formula


def test_agentic_v9_route_regressions_cover_all_routes_without_polluting_formal_scores() -> None:
    payload = _load(ROUTES_PATH)
    cases = payload["cases"]
    assert {case["expected_route"] for case in cases} == FORMAL_ROUTES
    assert any(case["expected_route"] == "graph_relational" and case["synthetic"] for case in cases)
    assert all(not case["include_in_formal_metrics"] for case in cases if case["synthetic"])
