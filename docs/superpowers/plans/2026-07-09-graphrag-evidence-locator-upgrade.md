# GraphRAG Evidence Locator Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild GraphRAG from graph-summary prompt injection into a provenance-aware graph-to-chunk evidence locator for academic literature QA.

**Architecture:** Keep the current NetworkX per-user graph and legacy graph mode as the rollback and ablation baseline. Add a parallel evidence-locator path that separates graph hints from source-backed evidence, expands provenance-backed graph items into original chunks/assets, packs those chunks with traceable graph boosts, and verifies answer claims against source evidence. Graph summaries and community summaries become retrieval/planning hints only.

**Tech Stack:** Python 3.13, FastAPI, Pydantic, NetworkX, LangChain/Gemini, FAISS/vector retrieval, pytest, ruff, React + TypeScript + Vite, Zustand, TanStack Query, Chakra UI, Vitest.

## Global Constraints

- Do not replace NetworkX with Neo4j or another graph database in this plan.
- Do not remove or rewrite `graph_raw_current`; it remains the legacy baseline, rollback path, and ablation condition.
- Do not allow raw graph summaries, community summaries, inferred relations, or edges without provenance to become final answer evidence.
- Final answers may cite only source chunks, tables, figures, formulas, captions, or verified evidence anchors.
- Every graph evidence item must expose `provenance_status = "full" | "partial" | "missing"`.
- Every resolved graph evidence item must also expose `resolution_status = "resolved" | "fuzzy_resolved" | "unresolved" | "stale"` and `verification_status = "quote_match" | "quote_mismatch" | "hash_mismatch" | "not_checked"`.
- Graph-to-chunk expansion must be traceable from query to graph item to source chunk to context pack to answer claim.
- Existing graph stores remain readable; legacy edges without provenance are marked `missing` rather than deleted.
- Graph auto gate must support manual override and must not be used as proof that GraphRAG itself improved.
- Exact numeric/table/formula questions may use graph as a locator only; graph must not directly answer exact extraction.
- Exact numeric/table/formula questions should use `locator_only` graph behavior when a source asset registry is available, and skip graph only when no usable graph/source-asset locator exists.
- Graph feature flags, graph snapshot version, graph schema version, and graph extraction prompt version must be stored with evaluation run snapshots.
- Graph debug and quality APIs must be auth-protected and per-user.
- Preserve router/service boundaries: router modules handle HTTP/auth/background scheduling; graph logic belongs in service/helper modules.
- Prompt changes must go through `prompts/graph_rag_prompts.json` and `core/prompt_loader.py`; do not add long prompt constants to production Python modules.
- If a route, API, persistence contract, or UI surface changes, update `docs/BACKEND.md`, `docs/generated/api-surface.md`, `Multimodal_RAG_System/docs/FRONTEND.md`, and `Multimodal_RAG_System/docs/generated/ui-surface.md` in the same change set.

---

## File Structure Map

### Backend Graph Core

- `graph_rag/schemas.py`: Graph evidence, hint, provenance, quality, debug, and API response models.
- `graph_rag/anchor_resolver.py`: Resolves graph provenance anchors back to source chunks/assets with hash and fuzzy fallback.
- `graph_rag/feature_flags.py`: Central flags for legacy, gated, locator, schema, alias, asset, quality, and debug paths.
- `graph_rag/store.py`: NetworkX store plus sidecars for provenance, aliases, schema, extraction runs, assets, quality, and version snapshots.
- `graph_rag/extractor.py`: Schema-first extraction output, evidence anchors, raw candidate buffer, and legacy fallback.
- `graph_rag/service.py`: Extraction orchestration, validation, canonicalization, quality computation, and snapshot save.
- `graph_rag/generic_mode.py`: Routing, `GraphHint`, `GraphEvidenceItem`, `GraphEvidenceBundle`, merge/gate decisions, and debug metadata.
- `graph_rag/local_search.py`: Local graph search producing structured evidence items.
- `graph_rag/global_search.py`: Global/community search producing `GraphHint` only unless source-backed evidence exists.
- `graph_rag/quality.py`: Quality metrics and actionable issue generation.
- `graph_rag/debug.py`: Query debugger service that returns entity linking, route, evidence, expansion, and context eligibility.
- `graph_rag/router.py`: Protected endpoints for quality/debug plus existing graph maintenance routes.

### Backend RAG, Evaluation, and Docs

- `data_base/RAG_QA_service.py`: New graph evidence bundle path, legacy wrapper, graph-to-chunk expansion, graph-boosted context packing, and graph gate.
- `data_base/context_packing.py`: Small focused helper for scoring graph-located chunks if no existing focused helper already owns this behavior.
- `evaluation/schemas.py`: Graph event/evidence row models and ablation flags.
- `evaluation/observability_storage.py`: Persistence for `evaluation_graph_events` and `evaluation_graph_evidence_items`, aligned with the evaluation dashboard observability storage track.
- `evaluation/analytics.py`: Graph metrics aggregation.
- `evaluation/rag_modes.py`: `graph_raw_current`, locator, claim-gated, planning-only, path-pruned, and router conditions.
- `tests/`: Focused backend regression tests listed per task.
- `docs/BACKEND.md`, `docs/generated/api-surface.md`, `docs/PRODUCT_SENSE.md`, `docs/RELIABILITY.md`: Backend documentation updates.

### Frontend Graph and Evaluation Surfaces

- `Multimodal_RAG_System/src/types/graph.ts`: Quality/debug/evidence response types.
- `Multimodal_RAG_System/src/services/graphApi.ts`: `/graph/quality` and `/graph/debug/search` clients.
- `Multimodal_RAG_System/src/hooks/useGraphData.ts`: Query hooks for quality/debug.
- `Multimodal_RAG_System/src/pages/GraphDemo.tsx`: Quality panels and query debugger.
- `Multimodal_RAG_System/src/stores/useSettingsStore.ts`: Default mode semantics and graph-auto settings.
- `Multimodal_RAG_System/src/types/rag.ts`: Graph mode/request typing if backend request shape changes.
- `Multimodal_RAG_System/src/pages/EvaluationCenter.tsx` and evaluation components: Graph ablation metrics where current analytics UI surfaces graph calls.
- `Multimodal_RAG_System/docs/FRONTEND.md`, `Multimodal_RAG_System/docs/generated/ui-surface.md`: Frontend documentation updates.

---

## Invariants

Use these invariants as hard review gates for every task:

```text
raw_graph_summary_is_not_final_evidence = true
community_summary_is_not_final_evidence = true
edge_without_provenance_is_not_final_context = true
unresolved_anchor_is_not_final_context = true
legacy_graph_mode_remains_available = true
graph_locator_failures_fall_back_to_vector_rag = true
graph_debug_routes_require_current_user = true
graph_feature_flags_are_stored_in_run_snapshot = true
graph_snapshot_version_is_stored_in_run_snapshot = true
exact_extraction_uses_locator_only_or_skip = true
```

---

## Task 1: Anchor Audit, Feature Flags, and Compatibility Contract

**Files:**
- Create: `graph_rag/feature_flags.py`
- Create: `graph_rag/anchor_resolver.py`
- Create: `tests/test_graph_anchor_contract.py`
- Modify: `graph_rag/schemas.py`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Produces: `GraphEvidenceMode`, `GraphFeatureFlags`, `get_graph_feature_flags() -> GraphFeatureFlags`
- Produces: `EvidenceAnchor`, `AnchorResolutionResult`, `ChunkAnchorResolver.resolve(anchor: EvidenceAnchor) -> AnchorResolutionResult`
- Consumes later: Tasks 2, 4, 5, 6, 8, and 10 use these flags and anchor models.

- [ ] **Step 1: Write failing tests for anchor fields and feature flag defaults**

Add `tests/test_graph_anchor_contract.py`:

```python
from graph_rag.feature_flags import get_graph_feature_flags
from graph_rag.schemas import EvidenceAnchor


def test_evidence_anchor_minimum_text_contract() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=3,
        page=7,
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        chunk_hash="chash",
        source_text_hash="sourcehash",
        confidence=0.91,
        extraction_model="gemini-test",
        extraction_prompt_version="graph-extract-v2",
    )

    assert anchor.doc_id == "doc-1"
    assert anchor.anchor_type == "text"
    assert anchor.provenance_status == "full"


def test_evidence_anchor_missing_quote_is_partial() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=3,
        confidence=0.72,
    )

    assert anchor.provenance_status == "partial"


def test_graph_feature_flags_default_to_legacy_safe_path() -> None:
    flags = get_graph_feature_flags({})

    assert flags.graph_raw_current_enabled is True
    assert flags.graph_evidence_locator_enabled is False
    assert flags.graph_provenance_gate_enabled is False
    assert flags.graph_to_chunk_enabled is False


def test_evidence_anchor_serializes_computed_provenance_status() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        chunk_hash="chash",
        confidence=0.91,
    )

    payload = anchor.model_dump(mode="json")

    assert payload["provenance_status"] == "full"


def test_graph_feature_flags_are_serializable_for_run_snapshot() -> None:
    flags = get_graph_feature_flags({"graph_to_chunk_enabled": "true"})

    assert flags.to_snapshot()["graph_to_chunk_enabled"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_anchor_contract.py -q
```

Expected: fails because `EvidenceAnchor`, `graph_rag.feature_flags`, and `ChunkAnchorResolver` do not exist.

- [ ] **Step 3: Add `EvidenceAnchor` and feature flags**

Add to `graph_rag/schemas.py`:

```python
from pydantic import computed_field


class EvidenceAnchor(BaseModel):
    doc_id: str
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    page: Optional[int] = None
    quote: Optional[str] = None
    quote_hash: Optional[str] = None
    chunk_hash: Optional[str] = None
    source_text_hash: Optional[str] = None
    markdown_char_start: Optional[int] = None
    markdown_char_end: Optional[int] = None
    asset_id: Optional[str] = None
    anchor_type: Literal["text", "table", "figure", "formula", "caption"] = "text"
    bbox: Optional[List[float]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_model: Optional[str] = None
    extraction_prompt_version: Optional[str] = None

    @computed_field
    @property
    def provenance_status(self) -> Literal["full", "partial", "missing"]:
        if not self.doc_id:
            return "missing"
        if self.chunk_id and self.quote and self.quote_hash and self.chunk_hash:
            return "full"
        if self.chunk_id or self.page is not None or self.asset_id:
            return "partial"
        return "missing"
```

Create `graph_rag/feature_flags.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class GraphFeatureFlags:
    graph_raw_current_enabled: bool = True
    graph_evidence_locator_enabled: bool = False
    graph_provenance_gate_enabled: bool = False
    graph_to_chunk_enabled: bool = False
    graph_auto_gate_enabled: bool = False
    graph_schema_v1_enabled: bool = False
    graph_alias_resolver_enabled: bool = False
    graph_asset_graph_enabled: bool = False
    graph_quality_dashboard_enabled: bool = False
    graph_debug_search_enabled: bool = False

    def to_snapshot(self) -> dict[str, bool]:
        return {
            "graph_raw_current_enabled": self.graph_raw_current_enabled,
            "graph_evidence_locator_enabled": self.graph_evidence_locator_enabled,
            "graph_provenance_gate_enabled": self.graph_provenance_gate_enabled,
            "graph_to_chunk_enabled": self.graph_to_chunk_enabled,
            "graph_auto_gate_enabled": self.graph_auto_gate_enabled,
            "graph_schema_v1_enabled": self.graph_schema_v1_enabled,
            "graph_alias_resolver_enabled": self.graph_alias_resolver_enabled,
            "graph_asset_graph_enabled": self.graph_asset_graph_enabled,
            "graph_quality_dashboard_enabled": self.graph_quality_dashboard_enabled,
            "graph_debug_search_enabled": self.graph_debug_search_enabled,
        }


def _flag(config: Mapping[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def get_graph_feature_flags(config: Mapping[str, object] | None = None) -> GraphFeatureFlags:
    source = config or {}
    return GraphFeatureFlags(
        graph_raw_current_enabled=_flag(source, "graph_raw_current_enabled", True),
        graph_evidence_locator_enabled=_flag(source, "graph_evidence_locator_enabled", False),
        graph_provenance_gate_enabled=_flag(source, "graph_provenance_gate_enabled", False),
        graph_to_chunk_enabled=_flag(source, "graph_to_chunk_enabled", False),
        graph_auto_gate_enabled=_flag(source, "graph_auto_gate_enabled", False),
        graph_schema_v1_enabled=_flag(source, "graph_schema_v1_enabled", False),
        graph_alias_resolver_enabled=_flag(source, "graph_alias_resolver_enabled", False),
        graph_asset_graph_enabled=_flag(source, "graph_asset_graph_enabled", False),
        graph_quality_dashboard_enabled=_flag(source, "graph_quality_dashboard_enabled", False),
        graph_debug_search_enabled=_flag(source, "graph_debug_search_enabled", False),
    )
```

Create `graph_rag/anchor_resolver.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from graph_rag.schemas import EvidenceAnchor


class ChunkLookup(Protocol):
    def by_chunk_id(self, user_id: str, chunk_id: str) -> Document | None: ...
    def by_doc_and_index(self, user_id: str, doc_id: str, chunk_index: int) -> Document | None: ...
    def by_chunk_hash(self, user_id: str, doc_id: str, chunk_hash: str) -> Document | None: ...
    def fuzzy_by_quote(self, user_id: str, doc_id: str, quote: str) -> Document | None: ...


@dataclass(frozen=True, slots=True)
class AnchorResolutionResult:
    anchor: EvidenceAnchor
    document: Document | None
    resolution_status: str
    verification_status: str
    reason: str


class ChunkAnchorResolver:
    def __init__(self, lookup: ChunkLookup) -> None:
        self._lookup = lookup

    def resolve(self, user_id: str, anchor: EvidenceAnchor) -> AnchorResolutionResult:
        if anchor.chunk_id:
            document = self._lookup.by_chunk_id(user_id, anchor.chunk_id)
            if document is not None:
                if anchor.chunk_hash and document.metadata.get("chunk_hash") != anchor.chunk_hash:
                    return AnchorResolutionResult(anchor, document, "stale", "hash_mismatch", "chunk_id_hash_mismatch")
                return AnchorResolutionResult(anchor, document, "resolved", "quote_match", "chunk_id")

        if anchor.chunk_index is not None:
            document = self._lookup.by_doc_and_index(user_id, anchor.doc_id, anchor.chunk_index)
            if document is not None:
                return AnchorResolutionResult(anchor, document, "resolved", "not_checked", "doc_id_chunk_index")

        if anchor.chunk_hash:
            document = self._lookup.by_chunk_hash(user_id, anchor.doc_id, anchor.chunk_hash)
            if document is not None:
                return AnchorResolutionResult(anchor, document, "resolved", "not_checked", "chunk_hash")

        if anchor.quote:
            document = self._lookup.fuzzy_by_quote(user_id, anchor.doc_id, anchor.quote)
            if document is not None:
                return AnchorResolutionResult(anchor, document, "fuzzy_resolved", "quote_match", "fuzzy_quote")

        return AnchorResolutionResult(anchor, None, "unresolved", "not_checked", "no_matching_chunk")
```

Add these resolver tests before closing Task 1:

```python
def test_anchor_resolver_detects_hash_mismatch(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_hash="old-hash",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "stale"
    assert result.verification_status == "hash_mismatch"


def test_anchor_resolver_fuzzy_quote_match(fake_lookup) -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        quote="MedSAM-2 uses memory attention.",
        quote_hash="qhash",
        confidence=0.8,
    )

    result = ChunkAnchorResolver(fake_lookup).resolve("user-1", anchor)

    assert result.resolution_status == "fuzzy_resolved"
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_anchor_contract.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag\schemas.py graph_rag\feature_flags.py graph_rag\anchor_resolver.py tests\test_graph_anchor_contract.py
```

Expected: tests pass and ruff reports no new issues.

- [ ] **Step 5: Commit**

```powershell
git add graph_rag\schemas.py graph_rag\feature_flags.py graph_rag\anchor_resolver.py tests\test_graph_anchor_contract.py docs\BACKEND.md
git commit -m "feat: define graph evidence anchor contract"
```

---

## Task 2: Graph Observability Baseline and Evaluation Tables

**Files:**
- Create: `tests/test_evaluation_graph_events.py`
- Modify: `evaluation/schemas.py`
- Modify: `evaluation/observability_storage.py`
- Modify: `evaluation/analytics.py`
- Modify: `graph_rag/generic_mode.py`
- Modify: `data_base/RAG_QA_service.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/generated/api-surface.md`

**Interfaces:**
- Produces: `EvaluationGraphEvent`, `EvaluationGraphEvidenceItem`
- Produces: repository methods `record_graph_event(...)` and `record_graph_evidence_items(...)`
- Consumes: Task 7 ablation and Task 12 frontend analytics.

- [ ] **Step 1: Write failing persistence tests**

Add `tests/test_evaluation_graph_events.py`:

```python
from evaluation.schemas import EvaluationGraphEvent, EvaluationGraphEvidenceItem


def test_graph_event_schema_records_route_and_success_rates() -> None:
    event = EvaluationGraphEvent(
        graph_event_id="ge-1",
        run_id="run-1",
        campaign_id="camp-1",
        span_id="span-1",
        graph_query="compare MedSAM and SAM-Med3D",
        graph_search_mode="generic",
        graph_evidence_mode="locator_to_chunk",
        graph_route="blended",
        router_reason="relation query with communities",
        graph_feature_flags={"graph_to_chunk_enabled": True},
        graph_snapshot_version="v003",
        graph_schema_version="graph-schema-v1",
        graph_extraction_prompt_version="graph-extract-v2",
        matched_entity_ids=["method:medsam"],
        community_ids=[3],
        node_count=2,
        edge_count=1,
        path_count=1,
        graph_latency_ms=42,
        graph_context_tokens=120,
        graph_to_chunk_success_rate=1.0,
        graph_noise_ratio=0.0,
    )

    assert event.graph_route == "blended"
    assert event.graph_evidence_mode == "locator_to_chunk"
    assert event.graph_feature_flags["graph_to_chunk_enabled"] is True
    assert event.graph_to_chunk_success_rate == 1.0


def test_graph_evidence_item_schema_tracks_context_lifecycle() -> None:
    item = EvaluationGraphEvidenceItem(
        graph_evidence_item_id="gei-1",
        graph_event_id="ge-1",
        node_ids=["method:medsam"],
        edge_ids=["edge-1"],
        relation_path=["method:medsam", "paper_proposes_method", "paper:medsam"],
        source_doc_ids=["doc-1"],
        source_chunk_ids=["chunk-1"],
        pages=[4],
        asset_ids=[],
        confidence=0.88,
        provenance_status="full",
        used_as_locator=True,
        packed_in_context=True,
        used_in_answer=False,
        supported_claim_ids=[],
    )

    assert item.provenance_status == "full"
    assert item.packed_in_context is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_graph_events.py -q
```

Expected: fails because the schemas do not exist.

- [ ] **Step 3: Add schemas and repository persistence**

Add to `evaluation/schemas.py`:

```python
class EvaluationGraphEvent(BaseModel):
    graph_event_id: str
    run_id: str
    campaign_id: str | None = None
    span_id: str | None = None
    graph_query: str
    graph_search_mode: str
    graph_evidence_mode: str = "raw_current"
    graph_route: str
    router_reason: str | None = None
    graph_feature_flags: dict[str, object] = Field(default_factory=dict)
    graph_snapshot_version: str | None = None
    graph_schema_version: str | None = None
    graph_extraction_prompt_version: str | None = None
    matched_entity_ids: list[str] = Field(default_factory=list)
    community_ids: list[int] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    path_count: int = 0
    graph_latency_ms: int | None = None
    graph_context_tokens: int = 0
    graph_to_chunk_success_rate: float | None = None
    graph_noise_ratio: float | None = None


class EvaluationGraphEvidenceItem(BaseModel):
    graph_evidence_item_id: str
    graph_event_id: str
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    relation_path: list[str] = Field(default_factory=list)
    source_doc_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    provenance_status: Literal["full", "partial", "missing"]
    used_as_locator: bool = True
    packed_in_context: bool = False
    used_in_answer: bool = False
    supported_claim_ids: list[str] = Field(default_factory=list)
```

In `evaluation/observability_storage.py`, add database creation/migration logic matching the existing evaluation observability storage style:

```python
CREATE_EVALUATION_GRAPH_EVENTS_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_graph_events (
    graph_event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    campaign_id TEXT,
    span_id TEXT,
    graph_query TEXT NOT NULL,
    graph_search_mode TEXT NOT NULL,
    graph_evidence_mode TEXT NOT NULL DEFAULT 'raw_current',
    graph_route TEXT NOT NULL,
    router_reason TEXT,
    graph_feature_flags_json TEXT NOT NULL DEFAULT '{}',
    graph_snapshot_version TEXT,
    graph_schema_version TEXT,
    graph_extraction_prompt_version TEXT,
    matched_entity_ids_json TEXT NOT NULL DEFAULT '[]',
    community_ids_json TEXT NOT NULL DEFAULT '[]',
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0,
    path_count INTEGER NOT NULL DEFAULT 0,
    graph_latency_ms INTEGER,
    graph_context_tokens INTEGER NOT NULL DEFAULT 0,
    graph_to_chunk_success_rate REAL,
    graph_noise_ratio REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""
```

Also add `evaluation_graph_evidence_items` with JSON text columns for list fields and repository methods that serialize lists with `json.dumps(..., ensure_ascii=False)`. Keep these methods in `EvaluationGraphEventRepository` and `EvaluationGraphEvidenceItemRepository` inside `evaluation/observability_storage.py`; do not introduce a second persistence layer named `evaluation/repository.py`.

- [ ] **Step 4: Emit graph observability events without changing behavior**

In `data_base/RAG_QA_service.py`, keep existing graph context behavior and add a best-effort call after `_get_graph_context(..., return_evidence=True)` when evaluation metadata is present. If evaluation metadata is absent, skip persistence and log debug-only fields.

Use a helper function:

```python
def _summarize_graph_evidence_for_log(evidence_units: list[GraphEvidence]) -> dict[str, object]:
    return {
        "node_count": sum(1 for item in evidence_units if item.evidence_type == "local_node"),
        "edge_count": sum(1 for item in evidence_units if item.evidence_type == "local_edge"),
        "community_count": sum(
            1 for item in evidence_units if item.evidence_type in {"community_summary", "community_answer"}
        ),
        "graph_context_tokens": sum(item.token_estimate for item in evidence_units),
    }
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_evaluation_graph_events.py tests/test_graphrag_integration.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation graph_rag\generic_mode.py data_base\RAG_QA_service.py tests\test_evaluation_graph_events.py
```

Expected: focused tests pass and legacy GraphRAG behavior remains unchanged.

- [ ] **Step 6: Commit**

```powershell
git add evaluation graph_rag\generic_mode.py data_base\RAG_QA_service.py tests\test_evaluation_graph_events.py docs\BACKEND.md docs\generated\api-surface.md
git commit -m "feat: record graph evaluation events"
```

---

## Task 3: Provenance Sidecars and Legacy Store Compatibility

**Files:**
- Create: `tests/test_graph_store_provenance_sidecars.py`
- Create: `tests/test_graph_legacy_store_compatibility.py`
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/store.py`
- Modify: `graph_rag/service.py`

**Interfaces:**
- Produces: `GraphEdgeProvenance`, `GraphExtractionRunManifest`
- Produces: `GraphStore.get_edge_provenance(edge_id: str) -> list[EvidenceAnchor]`
- Produces: `GraphStore.record_edge_provenance(edge_id: str, anchors: list[EvidenceAnchor]) -> None`
- Produces: minimal atomic sidecar writes using `*.tmp` files and `Path.replace()` before full snapshot versioning is implemented in Task 14.
- Consumes: Tasks 4, 5, 6, 9, and 10.

- [ ] **Step 1: Write failing store sidecar tests**

Add `tests/test_graph_store_provenance_sidecars.py`:

```python
from graph_rag.schemas import EvidenceAnchor
from graph_rag.store import GraphStore


def test_graph_store_persists_edge_provenance(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")

    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=0,
        page=1,
        quote="A source-backed relation.",
        quote_hash="quote-hash",
        chunk_hash="chunk-hash",
        confidence=0.9,
    )
    store.record_edge_provenance("edge:method:a:b", [anchor])
    store.save_sidecars()

    reloaded = GraphStore("user-1")
    anchors = reloaded.get_edge_provenance("edge:method:a:b")

    assert len(anchors) == 1
    assert anchors[0].provenance_status == "full"
```

Add `tests/test_graph_legacy_store_compatibility.py`:

```python
from graph_rag.store import GraphStore


def test_legacy_store_without_provenance_loads_as_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    source_id = store.add_node_from_extraction("MedSAM", "method", "doc-1")
    target_id = store.add_node_from_extraction("SAM", "method", "doc-1")
    store.add_edge_from_extraction(source_id, target_id, "extends", "doc-1")
    store.save()

    reloaded = GraphStore("user-1")
    edge_id = reloaded.edge_id(source_id, target_id, "extends")

    assert reloaded.get_edge_provenance(edge_id) == []
    assert reloaded.get_edge_provenance_status(edge_id) == "missing"


def test_edge_id_is_deterministic_hash_not_raw_relation_string(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")

    edge_id = store.edge_id("source/node", "target/node", "method reports metric")

    assert edge_id.startswith("edge:")
    assert " " not in edge_id
    assert "/" not in edge_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_store_provenance_sidecars.py tests/test_graph_legacy_store_compatibility.py -q
```

Expected: fails because provenance sidecar methods do not exist.

- [ ] **Step 3: Add provenance sidecars**

Add to `graph_rag/schemas.py`:

```python
class GraphEdgeProvenance(BaseModel):
    edge_id: str
    anchors: List[EvidenceAnchor] = Field(default_factory=list)
    extraction_run_id: Optional[str] = None
    schema_version: str = "graph-provenance-v1"
    extraction_prompt_version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class GraphExtractionRunManifest(BaseModel):
    extraction_run_id: str
    graph_extraction_version: str
    extractor_model: str | None = None
    prompt_version: str
    schema_version: str
    doc_id: str
    chunk_hashes: List[str] = Field(default_factory=list)
    temperature: float = 0.0
    validated: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
```

Add methods to `GraphStore`:

```python
from hashlib import sha1


def edge_id(self, source_id: str, target_id: str, relation: str) -> str:
    raw = f"{source_id}|{target_id}|{relation.strip().lower()}"
    return "edge:" + sha1(raw.encode("utf-8")).hexdigest()[:16]


def record_edge_provenance(self, edge_id: str, anchors: list[EvidenceAnchor]) -> None:
    self.edge_provenance[edge_id] = [anchor.model_dump(mode="json") for anchor in anchors]
    self.mark_dirty()


def get_edge_provenance(self, edge_id: str) -> list[EvidenceAnchor]:
    return [EvidenceAnchor(**item) for item in self.edge_provenance.get(edge_id, [])]


def get_edge_provenance_status(self, edge_id: str) -> Literal["full", "partial", "missing"]:
    anchors = self.get_edge_provenance(edge_id)
    if not anchors:
        return "missing"
    if any(anchor.provenance_status == "full" for anchor in anchors):
        return "full"
    if any(anchor.provenance_status == "partial" for anchor in anchors):
        return "partial"
    return "missing"
```

Persist `graph.provenance.json` in the same sidecar load/save style used by `graph.documents.json`, but use minimal atomic writes immediately:

```python
def _atomic_write_json(self, path: Path, payload: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with temp_path.open("rb") as handle:
        os.fsync(handle.fileno())
    temp_path.replace(path)
```

Use `_atomic_write_json(...)` for `graph.provenance.json`, `graph.aliases.json`, `graph.schema.json`, `graph.asset_links.json`, `graph.quality.json`, and `graph.extraction_runs.json` as those sidecars appear in later tasks. Task 14 still upgrades this to full versioned snapshots, but Task 3 must prevent graph/sidecar partial writes from the start.

- [ ] **Step 4: Ensure existing graph loads still pass**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graphrag_store_metadata.py tests/test_graph_store_provenance_sidecars.py tests/test_graph_legacy_store_compatibility.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add graph_rag\schemas.py graph_rag\store.py graph_rag\service.py tests\test_graph_store_provenance_sidecars.py tests\test_graph_legacy_store_compatibility.py
git commit -m "feat: persist graph edge provenance sidecars"
```

---

## Task 4: GraphHint and GraphEvidenceItem Separation

**Files:**
- Create: `tests/test_graph_evidence_gate.py`
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/generic_mode.py`
- Modify: `graph_rag/local_search.py`
- Modify: `graph_rag/global_search.py`

**Interfaces:**
- Produces: `GraphHint`, `GraphEvidenceItem`, `GraphEvidenceBundle`
- Produces: `merge_graph_evidence_bundle(...) -> GraphEvidenceBundle`
- Consumes: Tasks 5, 6, 10, and 12.

- [ ] **Step 1: Write failing gate tests**

Add `tests/test_graph_evidence_gate.py`:

```python
from graph_rag.generic_mode import merge_graph_evidence_bundle
from graph_rag.schemas import EvidenceAnchor, GraphEvidenceItem, GraphHint


def test_community_summary_is_hint_not_final_evidence() -> None:
    hint = GraphHint(
        hint_id="community:1",
        hint_type="community_summary",
        text="This community discusses SAM variants.",
        confidence=0.7,
        source_ids=["community:1"],
    )

    bundle = merge_graph_evidence_bundle(hints=[hint], evidence_items=[], token_budget=800)

    assert bundle.hints[0].usable_as_final_evidence is False
    assert bundle.final_context_items == []


def test_edge_without_provenance_is_excluded_from_final_context() -> None:
    item = GraphEvidenceItem(
        item_id="edge:1",
        graph_mode="local",
        source="edge",
        node_ids=["a", "b"],
        edge_ids=["edge:1"],
        source_chunk_ids=[],
        source_doc_ids=["doc-1"],
        pages=[],
        relation_type="extends",
        evidence_quote=None,
        summary="A extends B",
        confidence=0.8,
        provenance_status="missing",
        usable_as_context=False,
        use_reason="missing provenance",
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert bundle.final_context_items == []


def test_full_provenance_edge_can_enter_locator_bundle() -> None:
    anchor = EvidenceAnchor(
        doc_id="doc-1",
        chunk_id="chunk-1",
        chunk_index=1,
        quote="A extends B.",
        quote_hash="q",
        chunk_hash="c",
        confidence=0.9,
    )
    item = GraphEvidenceItem.from_anchor(
        item_id="edge:1",
        graph_mode="local",
        source="edge",
        edge_ids=["edge:1"],
        node_ids=["a", "b"],
        relation_type="extends",
        summary="A extends B",
        anchor=anchor,
    )

    bundle = merge_graph_evidence_bundle(hints=[], evidence_items=[item], token_budget=800)

    assert bundle.final_context_items[0].source_chunk_ids == ["chunk-1"]


def test_community_hint_never_becomes_final_context_even_with_high_confidence() -> None:
    hint = GraphHint(
        hint_id="community:important",
        hint_type="community_summary",
        text="A highly relevant community summary.",
        confidence=0.99,
        source_ids=["community:important"],
    )

    bundle = merge_graph_evidence_bundle(hints=[hint], evidence_items=[], token_budget=800)

    assert bundle.final_context_items == []
    assert bundle.hints[0].usable_as_final_evidence is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_gate.py -q
```

Expected: fails because the new bundle models do not exist.

- [ ] **Step 3: Add structured hint/evidence models**

Add to `graph_rag/schemas.py`:

```python
class GraphHint(BaseModel):
    hint_id: str
    hint_type: Literal["community_summary", "community_answer", "global_theme", "query_expansion"]
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_ids: List[str] = Field(default_factory=list)
    usable_as_final_evidence: bool = False


class GraphEvidenceItem(BaseModel):
    item_id: str
    graph_mode: Literal["local", "global", "blended"]
    source: Literal["edge", "node", "path", "asset"]
    node_ids: List[str] = Field(default_factory=list)
    edge_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    asset_ids: List[str] = Field(default_factory=list)
    relation_type: Optional[str] = None
    evidence_quote: Optional[str] = None
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance_status: Literal["full", "partial", "missing"]
    resolution_status: Literal["resolved", "fuzzy_resolved", "unresolved", "stale"] = "unresolved"
    verification_status: Literal["quote_match", "quote_mismatch", "hash_mismatch", "not_checked"] = "not_checked"
    usable_as_context: bool
    use_reason: str

    @classmethod
    def from_anchor(
        cls,
        *,
        item_id: str,
        graph_mode: Literal["local", "global", "blended"],
        source: Literal["edge", "node", "path", "asset"],
        edge_ids: List[str],
        node_ids: List[str],
        relation_type: Optional[str],
        summary: str,
        anchor: EvidenceAnchor,
        resolution_status: Literal["resolved", "fuzzy_resolved", "unresolved", "stale"] = "unresolved",
        verification_status: Literal["quote_match", "quote_mismatch", "hash_mismatch", "not_checked"] = "not_checked",
    ) -> "GraphEvidenceItem":
        usable = (
            anchor.provenance_status == "full"
            and resolution_status in {"resolved", "fuzzy_resolved"}
            and verification_status in {"quote_match", "not_checked"}
        )
        return cls(
            item_id=item_id,
            graph_mode=graph_mode,
            source=source,
            node_ids=node_ids,
            edge_ids=edge_ids,
            source_chunk_ids=[anchor.chunk_id] if anchor.chunk_id else [],
            source_doc_ids=[anchor.doc_id],
            pages=[anchor.page] if anchor.page is not None else [],
            asset_ids=[anchor.asset_id] if anchor.asset_id else [],
            relation_type=relation_type,
            evidence_quote=anchor.quote,
            summary=summary,
            confidence=anchor.confidence,
            provenance_status=anchor.provenance_status,
            resolution_status=resolution_status,
            verification_status=verification_status,
            usable_as_context=usable,
            use_reason="resolved provenance" if usable else "insufficient or unresolved provenance",
        )


class GraphEvidenceBundle(BaseModel):
    query: str
    route: str
    hints: List[GraphHint] = Field(default_factory=list)
    evidence_items: List[GraphEvidenceItem] = Field(default_factory=list)
    final_context_items: List[GraphEvidenceItem] = Field(default_factory=list)
    token_estimate: int = 0
```

- [ ] **Step 4: Add bundle merge function**

In `graph_rag/generic_mode.py`, keep `GraphEvidence` and `merge_graph_evidence(...)` for `graph_raw_current`. Add:

```python
def merge_graph_evidence_bundle(
    *,
    hints: list[GraphHint],
    evidence_items: list[GraphEvidenceItem],
    token_budget: int,
    query: str = "",
    route: str = "generic",
) -> GraphEvidenceBundle:
    selected: list[GraphEvidenceItem] = []
    spent_tokens = 0
    for item in sorted(evidence_items, key=lambda candidate: candidate.confidence, reverse=True):
        if (
            not item.usable_as_context
            or item.provenance_status != "full"
            or item.resolution_status not in {"resolved", "fuzzy_resolved"}
        ):
            continue
        estimate = estimate_token_count(item.summary or item.evidence_quote or "")
        if selected and spent_tokens + estimate > token_budget:
            continue
        selected.append(item)
        spent_tokens += estimate

    return GraphEvidenceBundle(
        query=query,
        route=route,
        hints=hints,
        evidence_items=evidence_items,
        final_context_items=selected,
        token_estimate=spent_tokens,
    )
```

- [ ] **Step 5: Update local/global search roles**

In `global_search.py`, map community summaries and generated community answers to `GraphHint`, not `GraphEvidenceItem`. In `local_search.py`, produce `GraphEvidenceItem` only when store provenance exists and the anchor resolver returns `resolved` or `fuzzy_resolved`; legacy text evidence remains available through current `local_search_evidence(...)`.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_gate.py tests/test_graphrag_integration.py tests/test_rag_graph_evidence_docs.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag tests\test_graph_evidence_gate.py
```

Expected: new gate tests pass; legacy graph context tests still pass.

- [ ] **Step 7: Commit**

```powershell
git add graph_rag tests\test_graph_evidence_gate.py
git commit -m "feat: separate graph hints from source-backed evidence"
```

---

## Task 5: Evidence-Locator Bundle and Legacy Wrapper in RAG QA

**Files:**
- Create: `tests/test_graph_evidence_bundle_wrapper.py`
- Modify: `data_base/RAG_QA_service.py`
- Modify: `graph_rag/generic_mode.py`
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Produces: `_get_graph_evidence_bundle(...) -> GraphEvidenceBundle`
- Keeps: `_get_graph_context(...) -> str | tuple[str, list[GraphEvidence]]` for legacy compatibility.
- Produces: `_get_graph_context_legacy_raw(...)` and `_render_graph_bundle_for_legacy_prompt(...)` with explicit names to avoid wrapper recursion.
- Consumes: Task 6 graph-to-chunk expansion.

- [ ] **Step 1: Write failing wrapper tests**

Add `tests/test_graph_evidence_bundle_wrapper.py`:

```python
from unittest.mock import AsyncMock, patch

import pytest

from data_base.RAG_QA_service import _get_graph_context, _get_graph_evidence_bundle
from graph_rag.schemas import GraphEvidenceBundle


@pytest.mark.asyncio
async def test_legacy_graph_context_wrapper_still_returns_string() -> None:
    bundle = GraphEvidenceBundle(query="q", route="local-first")

    with patch("data_base.RAG_QA_service._get_graph_evidence_bundle", new=AsyncMock(return_value=bundle)):
        context = await _get_graph_context("q", "user-1", search_mode="generic")

    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_new_graph_bundle_path_returns_structured_bundle() -> None:
    with patch("graph_rag.store.GraphStore") as store_cls:
        store = store_cls.return_value
        store.get_status.return_value.has_graph = False
        store.get_status.return_value.node_count = 0

        bundle = await _get_graph_evidence_bundle("q", "user-1", search_mode="generic")

    assert isinstance(bundle, GraphEvidenceBundle)
    assert bundle.final_context_items == []


@pytest.mark.asyncio
async def test_graph_context_wrapper_does_not_recurse(monkeypatch) -> None:
    calls = 0

    async def fake_legacy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "", []

    monkeypatch.setattr("data_base.RAG_QA_service._get_graph_context_legacy_raw", fake_legacy)

    await _get_graph_evidence_bundle("q", "user-1")

    assert calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_bundle_wrapper.py -q
```

Expected: fails because `_get_graph_evidence_bundle` does not exist.

- [ ] **Step 3: Add bundle path and keep legacy wrapper**

In `data_base/RAG_QA_service.py`, add:

```python
async def _get_graph_evidence_bundle(
    question: str,
    user_id: str,
    search_mode: str = "generic",
    graph_execution_hints: Optional[Dict[str, Any]] = None,
) -> GraphEvidenceBundle:
    from graph_rag.store import GraphStore

    store = GraphStore(user_id)
    status = store.get_status()
    if not status.has_graph or status.node_count == 0:
        return GraphEvidenceBundle(query=question, route="none")

    # Use existing route logic first, then swap evidence producer task-by-task.
    legacy_payload = await _get_graph_context_legacy_raw(
        question=question,
        user_id=user_id,
        search_mode=search_mode,
        graph_execution_hints=graph_execution_hints,
        return_evidence=True,
    )
    legacy_context, legacy_evidence = legacy_payload
    hints = []
    if legacy_context:
        hints.append(
            GraphHint(
                hint_id="legacy:graph-context",
                hint_type="query_expansion",
                text=legacy_context,
                confidence=0.5,
                source_ids=[item.evidence_id for item in legacy_evidence],
            )
        )
    return GraphEvidenceBundle(query=question, route=search_mode, hints=hints)
```

Then refactor `_get_graph_context` so the old implementation body becomes `_get_graph_context_legacy_raw(...)`. `_get_graph_context(...)` calls that legacy helper until `graph_evidence_locator_enabled` is enabled, and `_render_graph_bundle_for_legacy_prompt(...)` renders a bundle only for compatibility paths. Avoid recursion by placing the old body in `_get_graph_context_legacy_raw(...)` before adding the new bundle wrapper.

Add explicit mode tests:

```python
@pytest.mark.asyncio
async def test_graph_raw_current_uses_legacy_context(monkeypatch) -> None:
    called = False

    async def fake_legacy(*args, **kwargs):
        nonlocal called
        called = True
        return "=== Graph Evidence ===", []

    monkeypatch.setattr("data_base.RAG_QA_service._get_graph_context_legacy_raw", fake_legacy)

    context = await _get_graph_context("q", "user-1", search_mode="generic")

    assert called is True
    assert "Graph Evidence" in context
```

- [ ] **Step 4: Run integration tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_evidence_bundle_wrapper.py tests/test_graphrag_integration.py tests/test_rag_graph_evidence_docs.py -q
```

Expected: wrapper tests and legacy tests pass.

- [ ] **Step 5: Commit**

```powershell
git add data_base\RAG_QA_service.py graph_rag\generic_mode.py tests\test_graph_evidence_bundle_wrapper.py docs\BACKEND.md
git commit -m "feat: add graph evidence bundle wrapper"
```

---

## Task 6: Graph-to-Chunk Expansion and Graph-Boosted Context Packing

**Files:**
- Create: `data_base/context_packing.py`
- Create: `tests/test_graph_to_chunk_expansion.py`
- Create: `tests/test_graph_context_packing.py`
- Modify: `data_base/RAG_QA_service.py`
- Modify: `graph_rag/anchor_resolver.py`

**Interfaces:**
- Produces: `GraphLocatedChunk`
- Produces: `expand_graph_evidence_to_chunks(user_id: str, bundle: GraphEvidenceBundle, resolver: ChunkAnchorResolver) -> list[GraphLocatedChunk]`
- Produces: `score_graph_located_chunks(...) -> list[Document]`
- Consumes: Task 7 graph gate, Task 8 claim gate, Task 11 evaluation.

- [ ] **Step 1: Write failing expansion tests**

Add `tests/test_graph_to_chunk_expansion.py`:

```python
from langchain_core.documents import Document

from data_base.context_packing import GraphLocatedChunk, score_graph_located_chunks
from graph_rag.schemas import GraphEvidenceItem


def test_graph_located_chunk_preserves_selected_by_metadata() -> None:
    item = GraphEvidenceItem(
        item_id="edge-1",
        graph_mode="local",
        source="edge",
        node_ids=["a", "b"],
        edge_ids=["edge-1"],
        source_chunk_ids=["chunk-1"],
        source_doc_ids=["doc-1"],
        pages=[3],
        relation_type="extends",
        evidence_quote="A extends B.",
        summary="A extends B.",
        confidence=0.9,
        provenance_status="full",
        usable_as_context=True,
        use_reason="full provenance",
    )
    chunk = GraphLocatedChunk(
        document=Document(page_content="A extends B.", metadata={"chunk_id": "chunk-1", "doc_id": "doc-1"}),
        evidence_item=item,
        pre_boost_score=0.4,
    )

    scored = score_graph_located_chunks([chunk], required_modalities=[])

    assert scored[0].metadata["selected_by"] == "graph"
    assert scored[0].metadata["graph_boost_applied"] is True
    assert scored[0].metadata["graph_post_boost_score"] > scored[0].metadata["graph_pre_boost_score"]


def test_merge_vector_and_graph_docs_deduplicates_and_marks_both() -> None:
    vector_doc = Document(page_content="A extends B.", metadata={"chunk_id": "chunk-1", "selected_by": "vector"})
    graph_doc = Document(
        page_content="A extends B.",
        metadata={
            "chunk_id": "chunk-1",
            "selected_by": "graph",
            "graph_evidence_item_id": "edge-1",
        },
    )

    merged = merge_vector_and_graph_docs([vector_doc], [graph_doc], graph_chunk_ratio=0.35)

    assert len(merged) == 1
    assert merged[0].metadata["selected_by"] == "both"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_to_chunk_expansion.py -q
```

Expected: fails because `data_base.context_packing` does not exist.

- [ ] **Step 3: Add graph-located chunk scoring**

Create `data_base/context_packing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from graph_rag.schemas import GraphEvidenceItem


@dataclass(frozen=True, slots=True)
class GraphLocatedChunk:
    document: Document
    evidence_item: GraphEvidenceItem
    pre_boost_score: float = 0.0


def score_graph_located_chunks(
    chunks: list[GraphLocatedChunk],
    *,
    required_modalities: list[str],
    max_graph_chunks: int = 5,
) -> list[Document]:
    scored: list[Document] = []
    for chunk in chunks:
        score = chunk.pre_boost_score
        score += 0.08
        if chunk.evidence_item.provenance_status == "full":
            score += 0.05
        modality = str(chunk.document.metadata.get("modality", "text"))
        if modality in required_modalities:
            score += 0.07
        if chunk.evidence_item.confidence < 0.6:
            score -= 0.10

        metadata = dict(chunk.document.metadata)
        metadata.update(
            {
                "selected_by": "graph",
                "graph_evidence_item_id": chunk.evidence_item.item_id,
                "graph_boost_applied": True,
                "graph_pre_boost_score": chunk.pre_boost_score,
                "graph_post_boost_score": score,
                "graph_drop_reason": None,
                "provenance_status": chunk.evidence_item.provenance_status,
                "resolution_status": chunk.evidence_item.resolution_status,
            }
        )
        scored.append(Document(page_content=chunk.document.page_content, metadata=metadata))

    scored.sort(key=lambda doc: float(doc.metadata.get("graph_post_boost_score", 0.0)), reverse=True)
    return scored[:max_graph_chunks]
```

- [ ] **Step 4: Wire expansion into RAG path behind feature flag**

In `data_base/RAG_QA_service.py`, after vector retrieval and before context formatting:

```python
graph_located_docs: list[Document] = []
if enable_graph_rag and graph_feature_flags.graph_to_chunk_enabled:
    bundle = await _get_graph_evidence_bundle(
        question=question,
        user_id=user_id,
        search_mode=graph_search_mode,
        graph_execution_hints=graph_execution_hints,
    )
    graph_located_docs = await _expand_graph_bundle_to_documents(user_id, bundle)
    graph_located_docs = score_graph_located_chunks(
        graph_located_docs,
        required_modalities=_required_modalities_for_question(question),
    )
    docs = merge_vector_and_graph_docs(docs, graph_located_docs, graph_chunk_ratio=0.35)
```

Define `merge_vector_and_graph_docs(...)` with deduplication, hard caps, and interleaving. Do not prepend every graph chunk ahead of vector evidence.

```python
def merge_vector_and_graph_docs(
    vector_docs: list[Document],
    graph_docs: list[Document],
    *,
    graph_chunk_ratio: float,
    graph_every_n: int = 3,
) -> list[Document]:
    if not graph_docs:
        return vector_docs
    max_graph_docs = max(1, int((len(vector_docs) + len(graph_docs)) * graph_chunk_ratio))
    selected_graph_docs = graph_docs[:max_graph_docs]

    merged_by_chunk_id: dict[str, Document] = {}
    for doc in vector_docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if chunk_id:
            merged_by_chunk_id[chunk_id] = doc

    graph_only: list[Document] = []
    for doc in selected_graph_docs:
        chunk_id = str(doc.metadata.get("chunk_id", ""))
        if chunk_id and chunk_id in merged_by_chunk_id:
            existing = merged_by_chunk_id[chunk_id]
            metadata = {**existing.metadata, **doc.metadata, "selected_by": "both"}
            merged_by_chunk_id[chunk_id] = Document(page_content=existing.page_content, metadata=metadata)
        else:
            graph_only.append(doc)

    vector_after_dedup = list(merged_by_chunk_id.values())
    output: list[Document] = []
    graph_index = 0
    for index, doc in enumerate(vector_after_dedup, start=1):
        output.append(doc)
        if index % graph_every_n == 0 and graph_index < len(graph_only):
            output.append(graph_only[graph_index])
            graph_index += 1
    output.extend(graph_only[graph_index:])
    return output
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_to_chunk_expansion.py tests/test_graph_context_packing.py tests/test_rag_retrieval_logic.py -q
.\.venv\Scripts\python.exe -m ruff check data_base\context_packing.py data_base\RAG_QA_service.py tests\test_graph_to_chunk_expansion.py tests\test_graph_context_packing.py
```

Expected: graph-located chunks receive capped boost and legacy RAG tests pass.

- [ ] **Step 6: Commit**

```powershell
git add data_base\context_packing.py data_base\RAG_QA_service.py graph_rag\anchor_resolver.py tests\test_graph_to_chunk_expansion.py tests\test_graph_context_packing.py
git commit -m "feat: expand graph evidence into source chunks"
```

---

## Task 7: Graph Gate and Mode Semantics

**Files:**
- Create: `tests/test_graph_auto_gate.py`
- Modify: `data_base/RAG_QA_service.py`
- Modify: `evaluation/rag_modes.py`
- Modify: `Multimodal_RAG_System/src/stores/useSettingsStore.ts`
- Modify: `Multimodal_RAG_System/src/types/rag.ts`
- Modify: `Multimodal_RAG_System/src/pages/Chat.test.tsx`
- Modify: `Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `Multimodal_RAG_System/docs/generated/ui-surface.md`

**Interfaces:**
- Produces: `_classify_graph_need(question: str) -> GraphNeedDecision`
- Produces: graph modes `graph_raw_current`, `graph_locator_to_chunk`, `router_auto_graph`
- Consumes: Task 11 ablation.

- [ ] **Step 1: Write failing gate tests**

Add `tests/test_graph_auto_gate.py`:

```python
from data_base.RAG_QA_service import _classify_graph_need


def test_graph_gate_uses_graph_for_claim_scope() -> None:
    decision = _classify_graph_need("Compare the first claim scope of Weak-Mamba-UNet and Semi-Mamba-UNet")

    assert decision.use_graph is True
    assert decision.role == "locator"


def test_graph_gate_skips_graph_as_primary_for_exact_table_value() -> None:
    decision = _classify_graph_need(
        "What Params and FLOPs are reported in Table 1?",
        asset_registry_available=True,
    )

    assert decision.use_graph is True
    assert decision.role == "locator"
    assert decision.locator_only is True
    assert decision.final_graph_context_allowed is False
    assert "exact" in decision.reason.lower()


def test_graph_gate_skips_exact_table_value_when_no_locator_assets_exist() -> None:
    decision = _classify_graph_need(
        "What Params and FLOPs are reported in Table 1?",
        asset_registry_available=False,
    )

    assert decision.use_graph is False
    assert decision.role == "skip"


def test_graph_gate_allows_manual_override() -> None:
    decision = _classify_graph_need("What Params are reported?", manual_override=True)

    assert decision.use_graph is True
    assert decision.role == "locator"
    assert decision.locator_only is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_auto_gate.py -q
```

Expected: fails because `_classify_graph_need` does not exist.

- [ ] **Step 3: Add graph need classifier**

In `data_base/RAG_QA_service.py`:

```python
@dataclass(frozen=True, slots=True)
class GraphNeedDecision:
    use_graph: bool
    role: Literal["skip", "locator", "planning"]
    locator_only: bool
    final_graph_context_allowed: bool
    score: float
    reason: str


def _classify_graph_need(
    question: str,
    manual_override: bool = False,
    asset_registry_available: bool = False,
) -> GraphNeedDecision:
    normalized = question.lower()
    if manual_override:
        return GraphNeedDecision(True, "locator", True, False, 1.0, "manual override")

    exact_markers = ("table", "figure", "formula", "flops", "params", "exact", "數值", "公式", "表格")
    graph_markers = ("compare", "relationship", "claim", "scope", "contradict", "evolution", "跨文獻", "技術演進", "關係")

    if any(marker in normalized for marker in exact_markers):
        if asset_registry_available:
            return GraphNeedDecision(
                True,
                "locator",
                True,
                False,
                0.55,
                "exact extraction; graph may locate table/formula but cannot answer directly",
            )
        return GraphNeedDecision(False, "skip", False, False, 0.2, "exact extraction without usable graph asset locator")
    if any(marker in normalized for marker in graph_markers):
        return GraphNeedDecision(True, "locator", False, True, 0.8, "relationship or claim-scope query")
    return GraphNeedDecision(False, "skip", False, False, 0.3, "no graph-specific intent")
```

- [ ] **Step 4: Update evaluation modes**

In `evaluation/rag_modes.py`, add explicit conditions without removing existing `graph`:

```python
"graph_raw_current": {
    "enable_reranking": True,
    "enable_hyde": True,
    "enable_multi_query": True,
    "enable_graph_rag": True,
    "graph_search_mode": "generic",
    "graph_evidence_mode": "raw_current",
    "plain_mode": False,
},
"graph_locator_to_chunk": {
    "enable_reranking": True,
    "enable_hyde": True,
    "enable_multi_query": True,
    "enable_graph_rag": True,
    "graph_search_mode": "generic",
    "graph_evidence_mode": "locator_to_chunk",
    "plain_mode": False,
},
"router_auto_graph": {
    "enable_reranking": True,
    "enable_hyde": True,
    "enable_multi_query": True,
    "enable_graph_rag": True,
    "graph_search_mode": "generic",
    "graph_evidence_mode": "router_auto",
    "plain_mode": False,
},
```

- [ ] **Step 5: Update frontend default semantics**

In `useSettingsStore.ts`, change the default only after backend auto gate exists:

```ts
const DEFAULT_CHAT_MODE_ID: OfficialChatMode = 'advanced';
```

Rename UI description for graph:

```ts
description: 'Graph-assisted retrieval for cross-document relations and claim-scope questions.',
```

If a new `graph_auto` mode is preferred over changing `graph`, add it as an official preset and keep existing `graph` for explicit manual GraphRAG.

- [ ] **Step 6: Run backend and frontend focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_auto_gate.py tests/test_rag_modes_agentic.py tests/test_evaluation_pipeline.py -q
```

Then from `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx vitest run src/pages/Chat.test.tsx src/stores/useSettingsStore.test.ts
```

Expected: backend graph gate tests pass; frontend settings tests reflect new default semantics.

- [ ] **Step 7: Commit**

```powershell
git add data_base\RAG_QA_service.py evaluation\rag_modes.py tests\test_graph_auto_gate.py ..\Multimodal_RAG_System\src\stores\useSettingsStore.ts ..\Multimodal_RAG_System\src\types\rag.ts ..\Multimodal_RAG_System\src\pages\Chat.test.tsx ..\Multimodal_RAG_System\docs\FRONTEND.md ..\Multimodal_RAG_System\docs\generated\ui-surface.md
git commit -m "feat: gate graph usage by query intent"
```

---

## Task 8: Schema-First Extraction with Raw Candidate Buffer

**Files:**
- Create: `tests/test_graph_schema_first_extraction.py`
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/extractor.py`
- Modify: `prompts/graph_rag_prompts.json`
- Modify: `tests/test_graph_rag_prompts.py`

**Interfaces:**
- Produces: controlled node/edge schema constants.
- Produces: `RawGraphCandidate`
- Produces: extraction output with `canonical_name`, `aliases`, `entity_type`, `anchors`, `confidence`.
- Consumes: Task 9 canonicalization and Task 11 ablation.

- [ ] **Step 1: Write failing schema extraction tests**

Add `tests/test_graph_schema_first_extraction.py`:

```python
from graph_rag.extractor import classify_relation_for_answer_graph
from graph_rag.schemas import RawGraphCandidate


def test_allowed_relation_enters_answer_graph() -> None:
    result = classify_relation_for_answer_graph("method_evaluated_on_dataset")

    assert result.allowed is True
    assert result.normalized_relation == "method_evaluated_on_dataset"


def test_unknown_relation_goes_to_raw_candidate_buffer() -> None:
    result = classify_relation_for_answer_graph("loosely_related_to")

    assert result.allowed is False
    assert result.normalized_relation == "unknown_relation"


def test_raw_candidate_is_not_final_evidence() -> None:
    candidate = RawGraphCandidate(
        candidate_id="raw-1",
        candidate_type="unknown_relation",
        payload={"relation": "loosely_related_to"},
        source_doc_id="doc-1",
        confidence=0.5,
        needs_review=True,
    )

    assert candidate.needs_review is True
    assert candidate.usable_as_final_evidence is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_schema_first_extraction.py -q
```

Expected: fails because schema helper and raw candidate model do not exist.

- [ ] **Step 3: Add schema constants and raw candidate model**

In `graph_rag/schemas.py`:

```python
GRAPH_NODE_TYPES_V1 = {
    "Paper", "Method", "Model", "Dataset", "Metric", "Result", "Value",
    "Claim", "ClaimScope", "Limitation", "Task", "TrainingSetting",
    "PromptType", "ArchitectureComponent", "Ablation", "BenchmarkSetting",
    "EvidenceSpan", "Figure", "Table", "Formula",
}

GRAPH_EDGE_TYPES_V1 = {
    "paper_proposes_method", "method_uses_component", "method_evaluated_on_dataset",
    "method_reports_metric", "result_reports_value", "paper_contains_table",
    "paper_contains_figure", "paper_contains_formula", "claim_supported_by_evidence",
    "claim_contradicted_by_evidence", "method_compares_to_method",
    "method_requires_prompt", "method_supports_prompt_free",
    "method_uses_supervision", "claim_has_scope", "result_has_value",
    "table_reports_result", "formula_defines_variable", "method_has_training_setting",
    "claim_limited_to_setting",
}


class RawGraphCandidate(BaseModel):
    candidate_id: str
    candidate_type: str
    payload: Dict[str, object]
    source_doc_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = True
    usable_as_final_evidence: bool = False
```

In `graph_rag/extractor.py`:

```python
@dataclass(frozen=True, slots=True)
class RelationSchemaDecision:
    allowed: bool
    normalized_relation: str


def classify_relation_for_answer_graph(relation: str) -> RelationSchemaDecision:
    normalized = relation.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in GRAPH_EDGE_TYPES_V1:
        return RelationSchemaDecision(True, normalized)
    return RelationSchemaDecision(False, "unknown_relation")
```

- [ ] **Step 4: Update prompt registry**

Update `prompts/graph_rag_prompts.json` extraction prompt entries so the LLM is instructed to emit controlled node/edge schema values and evidence anchors. Keep wording scoped to extraction; do not rewrite unrelated prompts.

Add prompt test assertions in `tests/test_graph_rag_prompts.py`:

```python
def test_graph_extraction_prompt_mentions_provenance_and_schema() -> None:
    registry = get_graph_rag_prompt_registry()
    prompt = registry.format("one_pass_extraction", text="sample")

    assert "EvidenceAnchor" in prompt or "evidence anchor" in prompt.lower()
    assert "method_evaluated_on_dataset" in prompt
    assert "unknown_relation" in prompt
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_schema_first_extraction.py tests/test_graph_rag_prompts.py tests/test_graph_rag_extractor.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag\schemas.py graph_rag\extractor.py tests\test_graph_schema_first_extraction.py tests\test_graph_rag_prompts.py
```

Expected: schema tests pass and existing extractor tests still pass.

- [ ] **Step 6: Commit**

```powershell
git add graph_rag\schemas.py graph_rag\extractor.py prompts\graph_rag_prompts.json tests\test_graph_schema_first_extraction.py tests\test_graph_rag_prompts.py
git commit -m "feat: constrain graph extraction schema"
```

---

## Task 9: Alias Index and Claim-Scope-Safe Canonicalization

**Files:**
- Create: `tests/test_graph_alias_canonicalization.py`
- Modify: `graph_rag/store.py`
- Modify: `graph_rag/service.py`
- Modify: `graph_rag/entity_resolver.py`
- Modify: `graph_rag/generic_mode.py`
- Modify: `graph_rag/schemas.py`

**Interfaces:**
- Produces: `CanonicalEntity`, `ClaimIdentity`
- Produces: `GraphStore.find_canonical_node(label: str, entity_type: str | None) -> str | None`
- Produces sidecars: `graph.aliases.json`, `type_index`, `doc_index`
- Consumes: Task 10 quality/debug and Task 11 ablation.

- [ ] **Step 1: Write failing canonicalization tests**

Add `tests/test_graph_alias_canonicalization.py`:

```python
from graph_rag.schemas import ClaimIdentity
from graph_rag.store import GraphStore


def test_method_alias_resolves_to_canonical_node(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    canonical_id = store.upsert_canonical_entity(
        canonical_name="MedSAM-2",
        entity_type="Method",
        aliases=["MedSAM2", "MedSAM 2"],
        source_doc_ids=["doc-1"],
    )

    assert store.find_canonical_node("MedSAM2", "Method") == canonical_id


def test_claim_identity_requires_scope_and_source_doc() -> None:
    claim = ClaimIdentity(
        claim_type="first_claim",
        subject="Weak-Mamba-UNet",
        scope="scribble-based weakly supervised medical image segmentation",
        condition="scribble supervision",
        source_doc="weak-mamba.pdf",
    )

    assert "Weak-Mamba-UNet" in claim.stable_key
    assert "scribble" in claim.stable_key
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_alias_canonicalization.py -q
```

Expected: fails because alias and claim identity models do not exist.

- [ ] **Step 3: Add alias and claim models**

In `graph_rag/schemas.py`:

```python
class CanonicalEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    entity_type: str
    aliases: List[str] = Field(default_factory=list)
    source_doc_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    review_status: Literal["auto", "needs_review", "reviewed"] = "auto"


class ClaimIdentity(BaseModel):
    claim_type: str
    subject: str
    scope: str
    condition: Optional[str] = None
    source_doc: str

    @property
    def stable_key(self) -> str:
        parts = [self.claim_type, self.subject, self.scope, self.condition or "", self.source_doc]
        return "::".join(part.strip().lower() for part in parts if part.strip())
```

- [ ] **Step 4: Add store indexes**

Add `alias_index`, `type_index`, and `doc_index` sidecars in `GraphStore`. Do not automatically merge `Claim`, `Result`, `Table`, or `Formula`.

Use this guard:

```python
AUTO_MERGE_ENTITY_TYPES = {"Paper", "Dataset", "Metric", "Method"}
REVIEW_REQUIRED_ENTITY_TYPES = {"Model", "ArchitectureComponent", "TrainingSetting"}
NEVER_AUTO_MERGE_ENTITY_TYPES = {"Claim", "Result", "Table", "Formula"}
```

`REVIEW_REQUIRED_ENTITY_TYPES` must not be silently merged. For these types, create alias candidates with `review_status="needs_review"` unless there is an exact alias match inside the same `source_doc_ids` scope. This prevents variants such as `U-Mamba`, `U-Mamba_Enc`, and `U-Mamba_Bot` from collapsing into one node.

- [ ] **Step 5: Wire query entity linking**

In `generic_mode.py`, before local/global search:

```python
def link_query_entities(store: GraphStore, query_terms: list[str]) -> list[str]:
    linked: list[str] = []
    for term in query_terms:
        node_id = store.find_canonical_node(term, None)
        if node_id and node_id not in linked:
            linked.append(node_id)
    return linked
```

Keep existing vector/fuzzy search fallback when alias match is absent.

- [ ] **Step 6: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_alias_canonicalization.py tests/test_graph_local_search_vector.py tests/test_graphrag_store_metadata.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag tests\test_graph_alias_canonicalization.py
```

Expected: alias tests pass and local graph search remains functional.

- [ ] **Step 7: Commit**

```powershell
git add graph_rag tests\test_graph_alias_canonicalization.py
git commit -m "feat: add graph alias and claim-scope indexes"
```

---

## Task 10: Graph Quality API and Query Debugger

**Files:**
- Create: `graph_rag/quality.py`
- Create: `graph_rag/debug.py`
- Create: `tests/test_graph_quality_api.py`
- Create: `tests/test_graph_debug_search_api.py`
- Modify: `graph_rag/router.py`
- Modify: `graph_rag/schemas.py`
- Modify: `docs/BACKEND.md`
- Modify: `docs/generated/api-surface.md`

**Interfaces:**
- Produces: `GraphQualityResponse`
- Produces: `GraphRuntimeQualityResponse`
- Produces: `GraphDebugSearchResponse`
- Produces routes: `GET /graph/quality`, `GET /graph/runtime-quality`, `POST /graph/debug/search`
- Consumes: Task 12 frontend UI.

- [ ] **Step 1: Write failing API tests**

Add `tests/test_graph_quality_api.py`:

```python
from graph_rag.quality import compute_graph_quality, compute_graph_runtime_quality
from graph_rag.store import GraphStore


def test_static_quality_reports_graph_store_metrics(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    report = compute_graph_quality(store)

    assert report.num_nodes == 0
    assert report.num_edges == 0


def test_runtime_quality_flags_community_summary_used_as_evidence() -> None:
    report = compute_graph_runtime_quality(
        campaign_id="campaign-1",
        community_summary_used_as_evidence_count=1,
    )

    assert any(issue.code == "community_summary_used_as_evidence" for issue in report.issues)
```

Add `tests/test_graph_debug_search_api.py`:

```python
from graph_rag.debug import build_debug_search_response
from graph_rag.schemas import GraphEvidenceBundle


def test_debug_response_explains_final_context_eligibility() -> None:
    response = build_debug_search_response(
        query="Weak-Mamba first claim scope",
        bundle=GraphEvidenceBundle(query="Weak-Mamba first claim scope", route="local-first"),
        entity_links=[],
    )

    assert response.query == "Weak-Mamba first claim scope"
    assert response.route == "local-first"
    assert response.final_context_items == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_quality_api.py tests/test_graph_debug_search_api.py -q
```

Expected: fails because quality/debug modules do not exist.

- [ ] **Step 3: Add quality models and computation**

In `graph_rag/schemas.py`:

```python
class GraphQualityIssue(BaseModel):
    code: str
    severity: Literal["info", "warning", "critical"]
    message: str
    recommended_action: str


class GraphQualityResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    num_nodes: int
    num_edges: int
    edge_with_provenance_ratio: float
    generic_relation_ratio: float
    duplicate_method_node_ratio: float
    orphan_node_ratio: float
    graph_to_chunk_success_rate: float | None = None
    table_coverage_ratio: float | None = None
    figure_coverage_ratio: float | None = None
    formula_coverage_ratio: float | None = None
    claim_scope_missing_count: int = 0
    issues: List[GraphQualityIssue] = Field(default_factory=list)


class GraphRuntimeQualityResponse(BaseModel):
    campaign_id: Optional[str] = None
    community_summary_used_as_evidence_count: int = 0
    unsupported_graph_claim_rate: float | None = None
    graph_context_noise_ratio: float | None = None
    unresolved_anchor_count: int = 0
    graph_to_chunk_success_rate: float | None = None
    issues: List[GraphQualityIssue] = Field(default_factory=list)
```

Create `graph_rag/quality.py` with deterministic static calculations from the store; subtract score for each critical issue and return actionable messages. Runtime violation metrics such as `community_summary_used_as_evidence_count`, `unsupported_graph_claim_rate`, and `graph_context_noise_ratio` must come from evaluation graph events/evidence tables through `evaluation/analytics.py`, not from `GraphStore` alone.

- [ ] **Step 4: Add debug models and route**

In `graph_rag/schemas.py`:

```python
class GraphDebugSearchRequest(BaseModel):
    query: str
    search_mode: str = "generic"


class GraphDebugSearchResponse(BaseModel):
    query: str
    route: str
    entity_links: List[Dict[str, object]] = Field(default_factory=list)
    hints: List[GraphHint] = Field(default_factory=list)
    evidence_items: List[GraphEvidenceItem] = Field(default_factory=list)
    final_context_items: List[GraphEvidenceItem] = Field(default_factory=list)
```

In `graph_rag/router.py`, add protected endpoints using existing `Depends(get_current_user_id)` pattern.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_quality_api.py tests/test_graph_debug_search_api.py tests/test_graph_router_rebuild_full.py tests/test_router_boundaries.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag tests\test_graph_quality_api.py tests\test_graph_debug_search_api.py
```

Expected: quality/debug tests pass and router boundary tests remain green.

- [ ] **Step 6: Commit**

```powershell
git add graph_rag docs\BACKEND.md docs\generated\api-surface.md tests\test_graph_quality_api.py tests\test_graph_debug_search_api.py
git commit -m "feat: expose graph quality and debug search"
```

---

## Task 11: Multimodal Asset-Aware Graph Anchors

**Files:**
- Create: `tests/test_graph_asset_links.py`
- Modify: `graph_rag/schemas.py`
- Modify: `graph_rag/store.py`
- Modify: `graph_rag/extractor.py`
- Modify: `pdfserviceMD/service.py`
- Modify: `multimodal_rag/*` only where existing asset metadata is produced
- Modify: `docs/BACKEND.md`

**Interfaces:**
- Produces: `GraphAssetLink`
- Produces sidecar: `graph.asset_links.json`
- Produces: `GraphStore.record_asset_link(...)`, `GraphStore.get_asset_links_for_doc(...)`
- Consumes: Task 13 ablation for Q15/Q16.

- [ ] **Step 1: Write failing asset link tests**

Add `tests/test_graph_asset_links.py`:

```python
from graph_rag.schemas import GraphAssetLink
from graph_rag.store import GraphStore


def test_graph_asset_link_records_table_location(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    link = GraphAssetLink(
        asset_id="table-doc-1-1",
        doc_id="doc-1",
        page=5,
        asset_type="table",
        caption="Table 1. Params and FLOPs.",
        asset_text_hash="asset-hash",
        asset_parse_status="parsed",
        source_chunk_id="chunk-table-1",
    )

    store.record_asset_link(link)
    store.save_sidecars()

    reloaded = GraphStore("user-1")
    links = reloaded.get_asset_links_for_doc("doc-1")

    assert links[0].asset_type == "table"
    assert links[0].source_chunk_id == "chunk-table-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_asset_links.py -q
```

Expected: fails because `GraphAssetLink` and store methods do not exist.

- [ ] **Step 3: Add minimal asset registry contract**

In `graph_rag/schemas.py`:

```python
class GraphAssetLink(BaseModel):
    asset_id: str
    doc_id: str
    page: Optional[int] = None
    asset_type: Literal["table", "figure", "formula", "caption"]
    caption: Optional[str] = None
    text_or_markdown: Optional[str] = None
    asset_text_hash: Optional[str] = None
    asset_parse_status: Literal["parsed", "partial", "failed", "not_attempted"] = "not_attempted"
    bbox: Optional[List[float]] = None
    source_chunk_id: Optional[str] = None
```

In `GraphStore`, persist `graph.asset_links.json` and provide:

```python
def record_asset_link(self, link: GraphAssetLink) -> None:
    self.asset_links[link.asset_id] = link.model_dump(mode="json")
    self.mark_dirty()


def get_asset_links_for_doc(self, doc_id: str) -> list[GraphAssetLink]:
    return [
        GraphAssetLink(**item)
        for item in self.asset_links.values()
        if item.get("doc_id") == doc_id
    ]
```

- [ ] **Step 4: Connect asset links to evidence anchors**

When extractor sees asset metadata, create `EvidenceAnchor(anchor_type="table" | "figure" | "formula", asset_id=..., doc_id=..., page=...)`. Do not use graph asset links as direct answer text unless the source asset text or chunk is retrieved.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_asset_links.py tests/test_pdfservice_background_processing.py tests/test_visual_tool_trigger.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag pdfserviceMD multimodal_rag tests\test_graph_asset_links.py
```

Expected: asset links persist and existing PDF/multimodal tests remain green.

- [ ] **Step 6: Commit**

```powershell
git add graph_rag pdfserviceMD multimodal_rag tests\test_graph_asset_links.py docs\BACKEND.md
git commit -m "feat: link graph evidence to document assets"
```

---

## Task 12: Frontend Graph Quality and Query Debugger

**Files:**
- Modify: `Multimodal_RAG_System/src/types/graph.ts`
- Modify: `Multimodal_RAG_System/src/services/graphApi.ts`
- Modify: `Multimodal_RAG_System/src/hooks/useGraphData.ts`
- Modify: `Multimodal_RAG_System/src/pages/GraphDemo.tsx`
- Modify: `Multimodal_RAG_System/src/pages/GraphDemo.test.tsx`
- Modify: `Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `Multimodal_RAG_System/docs/generated/ui-surface.md`

**Interfaces:**
- Consumes: `GET /graph/quality`
- Consumes: `POST /graph/debug/search`
- Produces: quality panel with actionable issues and debugger table showing final-context eligibility.

- [ ] **Step 1: Write failing frontend API and UI tests**

In `src/pages/GraphDemo.test.tsx`, add assertions for quality and debugger:

```tsx
it('renders graph quality issues and query debugger controls', async () => {
  render(
    <TestProviders>
      <GraphDemo />
    </TestProviders>
  );

  expect(screen.getByText(/Graph Quality/i)).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: /Graph debug query/i })).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npx vitest run src/pages/GraphDemo.test.tsx
```

Expected: fails because UI and hooks do not exist.

- [ ] **Step 3: Add frontend types and API functions**

In `src/types/graph.ts`:

```ts
export interface GraphQualityIssue {
  code: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  recommended_action: string;
}

export interface GraphQualityResponse {
  score: number;
  num_nodes: number;
  num_edges: number;
  edge_with_provenance_ratio: number;
  generic_relation_ratio: number;
  duplicate_method_node_ratio: number;
  orphan_node_ratio: number;
  graph_to_chunk_success_rate?: number | null;
  table_coverage_ratio?: number | null;
  figure_coverage_ratio?: number | null;
  formula_coverage_ratio?: number | null;
  claim_scope_missing_count: number;
  community_summary_used_as_evidence_count: number;
  issues: GraphQualityIssue[];
}

export interface GraphDebugSearchRequest {
  query: string;
  search_mode?: GraphSearchMode;
}

export interface GraphDebugSearchResponse {
  query: string;
  route: string;
  entity_links: Array<Record<string, unknown>>;
  hints: Array<Record<string, unknown>>;
  evidence_items: Array<Record<string, unknown>>;
  final_context_items: Array<Record<string, unknown>>;
}
```

In `src/services/graphApi.ts`:

```ts
export async function getGraphQuality(): Promise<GraphQualityResponse> {
  const response = await api.get<GraphQualityResponse>('/graph/quality');
  return response.data;
}

export async function debugGraphSearch(
  request: GraphDebugSearchRequest
): Promise<GraphDebugSearchResponse> {
  const response = await api.post<GraphDebugSearchResponse>('/graph/debug/search', request);
  return response.data;
}
```

- [ ] **Step 4: Add hooks and page sections**

In `useGraphData.ts`, add `useGraphQuality()` and `useDebugGraphSearch()` using TanStack Query/mutation patterns already used in the file.

In `GraphDemo.tsx`, add a compact quality section above the graph visualization and a debugger section with:

```text
query input
search mode select
route result
entity links table
evidence items table
final context eligibility reason
eligible_for_final_context yes/no
source_chunk_resolution_status
verification_status
```

Keep the layout operational and dense; do not create a marketing-style section.

- [ ] **Step 5: Run frontend verification**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx tsc --noEmit
npx vitest run src/pages/GraphDemo.test.tsx src/hooks/useGraphData.test.tsx
```

Expected: lint, typecheck, and focused tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src\types\graph.ts src\services\graphApi.ts src\hooks\useGraphData.ts src\pages\GraphDemo.tsx src\pages\GraphDemo.test.tsx docs\FRONTEND.md docs\generated\ui-surface.md
git commit -m "feat: add graph quality and debug UI"
```

---

## Task 13: GraphRAG Ablation Campaign Conditions

**Files:**
- Create: `tests/test_graph_ablation_conditions.py`
- Modify: `evaluation/rag_modes.py`
- Modify: `evaluation/agentic_evaluation_service.py`
- Modify: `evaluation/analytics.py`
- Modify: `Multimodal_RAG_System/src/pages/EvaluationCenter.tsx`
- Modify: `Multimodal_RAG_System/src/components/evaluation/*` where graph metrics are displayed
- Modify: `docs/BACKEND.md`
- Modify: `docs/PRODUCT_SENSE.md`

**Interfaces:**
- Produces ablation groups:
  - Graph evidence ablation: `no_graph`, `graph_raw_current`, `graph_provenance_gated`, `graph_locator_to_chunk`, `graph_locator_claim_gate`
  - Graph usage policy: `always_no_graph`, `always_graph_locator`, `router_auto_graph`, `oracle_graph_router`
  - Graph query strategy: `local_first`, `global_first`, `blended`, `path_pruned`, `planning_only`

- [ ] **Step 1: Write failing ablation condition tests**

Add `tests/test_graph_ablation_conditions.py`:

```python
from evaluation.rag_modes import RAG_MODES


def test_graph_evidence_ablation_conditions_exist() -> None:
    for mode in [
        "graph_raw_current",
        "graph_provenance_gated",
        "graph_locator_to_chunk",
        "graph_locator_claim_gate",
    ]:
        assert mode in RAG_MODES
        assert RAG_MODES[mode]["enable_graph_rag"] is True


def test_router_auto_graph_is_separate_from_graph_component_ablation() -> None:
    assert RAG_MODES["router_auto_graph"]["graph_evidence_mode"] == "router_auto"
    assert RAG_MODES["router_auto_graph"]["ablation_family"] == "graph_usage_policy"
    assert RAG_MODES["graph_locator_to_chunk"]["graph_evidence_mode"] == "locator_to_chunk"
    assert RAG_MODES["graph_locator_to_chunk"]["ablation_family"] == "graph_evidence"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_ablation_conditions.py -q
```

Expected: fails because all ablation conditions are not present yet.

- [ ] **Step 3: Add condition definitions**

In `evaluation/rag_modes.py`, add the conditions from this task's interface. Keep `graph` as compatibility alias if existing UI/tests depend on it.

Use explicit `graph_evidence_mode` and `ablation_family` values:

```python
"graph_provenance_gated": {
    "enable_reranking": True,
    "enable_hyde": True,
    "enable_multi_query": True,
    "enable_graph_rag": True,
    "graph_search_mode": "generic",
    "graph_evidence_mode": "provenance_gated",
    "ablation_family": "graph_evidence",
    "plain_mode": False,
},
```

- [ ] **Step 4: Add analytics separation**

In `evaluation/analytics.py`, aggregate Graph evidence ablation separately from router policy ablation by `ablation_family`. Do not average `graph_evidence`, `graph_usage_policy`, and `graph_query_strategy` into one "GraphRAG improved" number.

Expose metrics:

```text
graph_node_hit_at_k
graph_edge_hit_at_k
graph_doc_hit_rate
graph_evidence_chunk_hit_rate
graph_to_chunk_success_rate
graph_context_noise_ratio
unsupported_graph_claim_rate
router_skip_graph_accuracy
router_use_graph_accuracy
```

- [ ] **Step 5: Run backend verification**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_ablation_conditions.py tests/test_evaluation_pipeline.py tests/test_evaluation_analytics_api.py -q
.\.venv\Scripts\python.exe -m ruff check evaluation tests\test_graph_ablation_conditions.py
```

Expected: ablation conditions exist and analytics tests pass.

- [ ] **Step 6: Run frontend evaluation verification**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx vitest run src/pages/EvaluationCenter.test.tsx src/components/evaluation/AgentBehaviorTab.test.tsx
```

Expected: frontend evaluation surface renders graph metrics without type errors.

- [ ] **Step 7: Commit**

```powershell
git add evaluation tests\test_graph_ablation_conditions.py docs\BACKEND.md docs\PRODUCT_SENSE.md ..\Multimodal_RAG_System\src\pages\EvaluationCenter.tsx ..\Multimodal_RAG_System\src\components\evaluation
git commit -m "feat: add graph evidence ablation conditions"
```

---

## Task 14: Versioned Graph Snapshots and Atomic Save

**Files:**
- Create: `tests/test_graph_snapshot_atomic_save.py`
- Modify: `graph_rag/store.py`
- Modify: `graph_rag/maintenance.py`
- Modify: `graph_rag/service.py`

**Interfaces:**
- Produces: versioned graph store layout:
  - `versions/v001/graph.pkl`
  - `versions/v001/graph.meta.json`
  - `versions/v001/graph.provenance.json`
  - `current.json`
- Produces: `GraphStore.save_snapshot() -> str`
- Produces: read-only load from current snapshot for query paths.

- [ ] **Step 1: Write failing snapshot tests**

Add `tests/test_graph_snapshot_atomic_save.py`:

```python
from graph_rag.store import GraphStore


def test_graph_snapshot_save_updates_current_after_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    store.add_node_from_extraction("MedSAM", "method", "doc-1")

    version = store.save_snapshot()

    assert version.startswith("v")
    assert (tmp_path / "user-1" / "current.json").exists()
    assert store.load_current_pointer()["current_version"] == version
    assert "sidecar_hashes" in store.load_current_pointer()


def test_query_loads_current_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("graph_rag.store.GRAPH_STORAGE_DIR", tmp_path)
    store = GraphStore("user-1")
    store.add_node_from_extraction("MedSAM", "method", "doc-1")
    store.save_snapshot()

    reloaded = GraphStore("user-1")

    assert reloaded.get_status().node_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_snapshot_atomic_save.py -q
```

Expected: fails because snapshot methods do not exist.

- [ ] **Step 3: Add snapshot save with validation**

In `GraphStore`, implement:

```python
def save_snapshot(self) -> str:
    version = self._next_snapshot_version()
    target = self._version_dir(version)
    temp = self._version_dir(f".{version}.tmp")
    temp.mkdir(parents=True, exist_ok=False)
    self._write_graph_files(temp)
    self._validate_snapshot(temp)
    temp.rename(target)
    self._write_current_pointer(
        {
            "current_version": version,
            "created_at": datetime.now().isoformat(),
            "schema_version": self.schema_version,
            "graph_hash": self._hash_file(target / "graph.pkl"),
            "sidecar_hashes": self._hash_snapshot_sidecars(target),
        }
    )
    return version
```

Use Windows-safe `Path.replace` for final pointer replacement. `current.json` must be written via temp file then `Path.replace(...)` and should look like:

```json
{
  "current_version": "v003",
  "created_at": "2026-07-09T12:00:00",
  "schema_version": "graph-schema-v1",
  "graph_hash": "sha256:...",
  "sidecar_hashes": {
    "graph.pkl": "sha256:...",
    "graph.provenance.json": "sha256:...",
    "graph.aliases.json": "sha256:..."
  }
}
```

Do not delete previous snapshots during this task.

- [ ] **Step 4: Switch rebuild to write new snapshot before swap**

In `graph_rag/service.py` and `graph_rag/maintenance.py`, use `save_snapshot()` after extraction/rebuild succeeds. Query paths continue to load `current`.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_snapshot_atomic_save.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag tests\test_graph_snapshot_atomic_save.py
```

Expected: snapshot tests pass and rebuild routes still pass.

- [ ] **Step 6: Commit**

```powershell
git add graph_rag tests\test_graph_snapshot_atomic_save.py
git commit -m "feat: save graph snapshots atomically"
```

---

## Task 15: Documentation and Final Verification

**Files:**
- Modify: `docs/BACKEND.md`
- Modify: `docs/generated/api-surface.md`
- Modify: `docs/PRODUCT_SENSE.md`
- Modify: `docs/RELIABILITY.md`
- Modify: `Multimodal_RAG_System/docs/FRONTEND.md`
- Modify: `Multimodal_RAG_System/docs/generated/ui-surface.md`
- Modify: `agent.md` only if a project-wide learned rule is added after an incident or correction.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: documented Graph Evidence Locator behavior, rollback behavior, flags, APIs, UI surfaces, and ablation interpretation.

- [ ] **Step 1: Update backend docs**

Add this wording to `docs/BACKEND.md`:

```markdown
GraphRAG is now split into a legacy raw-context baseline and a provenance-aware evidence locator. The evidence locator treats graph paths, community summaries, and global themes as retrieval hints. Final answer evidence must come from source chunks, tables, figures, formulas, captions, or full provenance anchors.
```

Add `/graph/quality` and `/graph/debug/search` to `docs/generated/api-surface.md`.

- [ ] **Step 2: Update frontend docs**

Add this wording to `Multimodal_RAG_System/docs/FRONTEND.md`:

```markdown
The Graph Workspace includes quality diagnostics and query debugging. It shows provenance coverage, relation quality, duplicate candidates, graph-to-chunk resolution, and whether graph items are eligible for final context.
```

Update `Multimodal_RAG_System/docs/generated/ui-surface.md` for Graph quality and debugger panels.

- [ ] **Step 3: Run backend focused verification**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graph_anchor_contract.py tests/test_evaluation_graph_events.py tests/test_graph_store_provenance_sidecars.py tests/test_graph_legacy_store_compatibility.py tests/test_graph_evidence_gate.py tests/test_graph_evidence_bundle_wrapper.py tests/test_graph_to_chunk_expansion.py tests/test_graph_context_packing.py tests/test_graph_auto_gate.py tests/test_graph_schema_first_extraction.py tests/test_graph_alias_canonicalization.py tests/test_graph_quality_api.py tests/test_graph_debug_search_api.py tests/test_graph_asset_links.py tests/test_graph_ablation_conditions.py tests/test_graph_snapshot_atomic_save.py -q
```

Expected: all Graph Evidence Locator focused tests pass.

- [ ] **Step 4: Run backend broader production-focused verification**

Run:

```powershell
$env:TEST_MODE='true'; $env:USE_FAKE_PROVIDERS='true'; .\.venv\Scripts\python.exe -m pytest tests/test_graphrag_integration.py tests/test_graph_rag_extractor.py tests/test_graphrag_store_metadata.py tests/test_graph_local_search_vector.py tests/test_graph_router_rebuild_full.py tests/test_graph_router_copy.py tests/test_rag_retrieval_logic.py tests/test_rag_graph_evidence_docs.py tests/test_evaluation_pipeline.py tests/test_evaluation_analytics_api.py tests/test_router_boundaries.py -q
.\.venv\Scripts\python.exe -m ruff check graph_rag data_base evaluation tests
```

Expected: production-focused backend tests pass and ruff reports no new issues.

- [ ] **Step 5: Run frontend verification**

From `D:\flutterserver\Multimodal_RAG_System`:

```powershell
npm run lint:ci
npx tsc --noEmit
npx vitest run src/pages/GraphDemo.test.tsx src/hooks/useGraphData.test.tsx src/pages/Chat.test.tsx src/pages/EvaluationCenter.test.tsx
npm run build
```

Expected: lint, typecheck, focused tests, and production build pass.

- [ ] **Step 6: Run legacy migration smoke**

Run a migration smoke before any ablation:

```text
1. Load a legacy graph store without graph.provenance.json.
2. Verify legacy edges report provenance_status = missing.
3. Verify graph_raw_current still returns legacy graph context.
4. Verify graph_locator_to_chunk returns no final graph evidence for unresolved legacy edges and falls back to vector RAG.
5. Rebuild one document into a new graph snapshot.
6. Verify locator path starts producing GraphEvidenceItem rows with graph_snapshot_version and graph_feature_flags_snapshot.
```

Expected: legacy stores do not crash, missing provenance never enters final context, and rebuilt graph snapshots produce source-backed evidence where anchors resolve.

- [ ] **Step 7: Run smoke ablation**

Run the smallest deterministic graph-focused campaign available in this repo. If the project lacks a ready CLI for this exact set, create the campaign through the existing evaluation API/UI using:

```text
Graph-suitable subset: Q2, Q3, Q7, Q8, Q11, Q13, Q14
Negative/exact subset: Q4, Q15, Q16
Modes: no_graph, graph_raw_current, graph_provenance_gated, graph_locator_to_chunk, router_auto_graph
Repeats: 1 for smoke, 3 for research run
```

Expected: campaign records graph events and graph evidence items for graph-enabled modes; exact subset records router skip decisions for `router_auto_graph`.

- [ ] **Step 8: Run execution readiness checklist**

Before closing the implementation branch, verify:

```text
[ ] legacy graph_raw_current still works
[ ] graph locator flags default off
[ ] graph debug routes require current user
[ ] community summaries never appear in final_context_items
[ ] edge without provenance never appears in final_context_items
[ ] unresolved or stale anchors are logged and skipped
[ ] graph-located chunks have selected_by metadata
[ ] duplicated vector/graph chunks are merged as selected_by = both
[ ] graph chunks are interleaved or capped rather than prepended without limit
[ ] graph feature flags are stored in run snapshot
[ ] graph snapshot version is stored in run snapshot
[ ] exact extraction queries are locator-only or skipped
[ ] graph body ablation and router policy ablation are separate
[ ] static graph quality and runtime graph violation metrics are reported separately
```

- [ ] **Step 9: Commit documentation and verification updates**

```powershell
git add docs ..\Multimodal_RAG_System\docs
git commit -m "docs: document graph evidence locator workflow"
```

---

## Self-Review

### Spec Coverage

- Phase -1 anchor and compatibility audit: Task 1.
- Serialized `provenance_status`, `resolution_status`, `verification_status`, hash mismatch, and fuzzy quote resolution: Task 1.
- Observability and graph-specific normalized tables: Task 2.
- Graph feature flags, graph evidence mode, graph snapshot/schema/prompt versions in run snapshots: Task 2 and Task 13.
- Provenance sidecars, minimal atomic sidecar writes, deterministic hashed edge IDs, and legacy compatibility: Task 3.
- `GraphHint` vs `GraphEvidenceItem`: Task 4.
- `_get_graph_context()` compatibility wrapper and structured bundle: Task 5.
- Graph-to-chunk expansion, capped graph boost, deduplication, and vector/graph interleaving: Task 6.
- Graph gate, frontend mode semantics, manual override, and exact extraction locator-only behavior: Task 7.
- Schema-first extraction with raw candidate buffer: Task 8.
- Alias merge and claim-scope-safe canonicalization: Task 9.
- Static graph quality, runtime graph quality, and query debugger: Task 10.
- Multimodal asset-aware graph anchors with asset parse status/hash: Task 11.
- Frontend quality/debug UI with final-context eligibility, resolution, and verification status: Task 12.
- Graph body ablation, router policy ablation, query strategy ablation, and `ablation_family`: Task 13.
- Versioned graph snapshots, atomic save, current pointer metadata, and sidecar hashes: Task 14.
- Docs, legacy migration smoke, execution readiness checklist, and final verification: Task 15.

### Placeholder Scan

The plan avoids unresolved placeholder language and gives concrete file paths, interfaces, test examples, commands, and expected outcomes for each task.

### Type Consistency

- `EvidenceAnchor` is introduced before `GraphEvidenceItem.from_anchor`.
- `GraphEvidenceBundle` is introduced before `_get_graph_evidence_bundle`.
- `GraphLocatedChunk` consumes `GraphEvidenceItem`.
- `GraphEvidenceItem.usable_as_context` depends on provenance plus resolution state.
- `GraphQualityResponse`, `GraphRuntimeQualityResponse`, and `GraphDebugSearchResponse` are used by backend routes before frontend clients consume them.
- `graph_evidence_mode` and `ablation_family` values in evaluation are explicit and not overloaded with the existing `mode` enum.
