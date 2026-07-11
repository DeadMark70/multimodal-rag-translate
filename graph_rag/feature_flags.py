from __future__ import annotations

from dataclasses import asdict, dataclass
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
        return asdict(self)


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
