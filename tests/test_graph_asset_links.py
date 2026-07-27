from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

from graph_rag.schemas import GraphAssetLink
from graph_rag.store import GraphStore


def test_graph_asset_link_persists_table_location(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    link = GraphAssetLink(
        asset_id="table-doc-1-1",
        doc_id="doc-1",
        page=5,
        asset_type="table",
        caption="Table 1. Params and FLOPs.",
        text_or_markdown="| Params | FLOPs |\n| --- | --- |\n| 4M | 10G |",
        asset_text_hash="asset-hash",
        asset_parse_status="parsed",
        source_chunk_id="chunk-table-1",
    )

    store.record_asset_link(link)
    store.save_sidecars()

    reloaded = GraphStore("user-1", storage_dir=tmp_path)
    links = reloaded.get_asset_links_for_doc("doc-1")

    assert links == [link]


def test_markdown_asset_parser_keeps_page_and_source_text() -> None:
    from graph_rag.assets import extract_markdown_asset_links

    links = extract_markdown_asset_links(
        doc_id="doc-1",
        markdown_text="""[[PAGE_5]]
Table 1. Params and FLOPs.
| Params | FLOPs |
| --- | --- |
| 4M | 10G |

$$\\mathcal{L} = \\mathcal{L}_{dice}$$
""",
    )

    table = next(link for link in links if link.asset_type == "table")
    formula = next(link for link in links if link.asset_type == "formula")
    caption = next(link for link in links if link.asset_type == "caption")

    assert table.page == 5
    assert table.asset_parse_status == "parsed"
    assert "4M" in table.text_or_markdown
    assert formula.text_or_markdown == "$$\\mathcal{L} = \\mathcal{L}_{dice}$$"
    assert caption.caption == "Table 1. Params and FLOPs."


def test_asset_probe_requires_a_matching_parsed_document_asset(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    store.record_asset_link(
        GraphAssetLink(
            asset_id="table-doc-1-1",
            doc_id="doc-1",
            asset_type="table",
            text_or_markdown="| A | B |",
            asset_text_hash="asset-hash",
            asset_parse_status="parsed",
        )
    )

    assert store.has_usable_asset_links({"doc-1"}, {"table"}) is True
    assert store.has_usable_asset_links({"doc-2"}, {"table"}) is False
    assert store.has_usable_asset_links({"doc-1"}, {"formula"}) is False


def test_visual_asset_link_uses_the_indexed_summary_chunk() -> None:
    from graph_rag.assets import build_visual_asset_links

    element = SimpleNamespace(
        id="visual-1",
        type="figure",
        page_number=3,
        bbox=[1, 2, 3, 4],
        summary="Figure summary extracted from the source image.",
        context_text="Figure 2 shows the architecture.",
        figure_reference="Figure 2",
    )

    links = build_visual_asset_links(doc_id="doc-1", elements=[element])

    assert links[0].asset_type == "figure"
    assert links[0].source_chunk_id == f"graph:asset:{links[0].asset_id}"
    assert element.asset_id == links[0].asset_id


def test_graph_asset_link_round_trips_resolvable_optional_metadata(tmp_path) -> None:
    link = GraphAssetLink(
        asset_id="figure-doc-1-1",
        doc_id="doc-1",
        page=2,
        asset_type="figure",
        caption="Figure 1",
        storage_reference="user-1/doc-1/page-2.png",
        sha256="a" * 64,
        width=640,
        height=480,
        printed_page_label="A-2",
        formula_id="Equation 4",
    )
    store = GraphStore("user-1", storage_dir=tmp_path)

    store.record_asset_link(link)
    store.save_sidecars()

    reloaded = GraphStore("user-1", storage_dir=tmp_path)
    assert reloaded.get_asset_links_for_doc("doc-1") == [link]
    payload = json.loads((tmp_path / "graph.asset_links.json").read_text("utf-8"))
    serialized = json.dumps(payload)
    assert "user-1/doc-1/page-2.png" in serialized
    assert "base64" not in serialized
    assert "data:image" not in serialized


def test_graph_asset_link_loads_legacy_sidecar_without_resolvable_metadata(
    tmp_path,
) -> None:
    (tmp_path / "graph.asset_links.json").write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "asset_id": "legacy-table",
                        "doc_id": "doc-1",
                        "page": 4,
                        "asset_type": "table",
                        "caption": "Table 2",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    links = GraphStore("user-1", storage_dir=tmp_path).get_asset_links_for_doc("doc-1")

    assert len(links) == 1
    assert links[0].storage_reference is None
    assert links[0].sha256 is None
    assert links[0].width is None
    assert links[0].height is None
    assert links[0].printed_page_label is None
    assert links[0].formula_id is None


def test_asset_lookup_is_authorized_locator_bound_and_bounded(tmp_path) -> None:
    store = GraphStore("user-1", storage_dir=tmp_path)
    for link in (
        GraphAssetLink(
            asset_id="figure-1",
            doc_id="doc-1",
            page=2,
            asset_type="figure",
            caption="Figure 1",
        ),
        GraphAssetLink(
            asset_id="table-1",
            doc_id="doc-1",
            page=3,
            asset_type="table",
            caption="Table 1",
        ),
        GraphAssetLink(
            asset_id="formula-1",
            doc_id="doc-1",
            page=4,
            asset_type="formula",
            formula_id="Equation 4",
        ),
        GraphAssetLink(
            asset_id="cross-document",
            doc_id="doc-2",
            page=2,
            asset_type="figure",
            caption="Figure 1",
        ),
    ):
        store.record_asset_link(link)

    assert [
        link.asset_id
        for link in store.lookup_asset_links(
            authorized_doc_ids={"doc-1"},
            page=2,
            figure_id="Figure 1",
            limit=1,
        )
    ] == ["figure-1"]
    assert [
        link.asset_id
        for link in store.lookup_asset_links(
            authorized_doc_ids={"doc-1"},
            table_id="Table 1",
        )
    ] == ["table-1"]
    assert [
        link.asset_id
        for link in store.lookup_asset_links(
            authorized_doc_ids={"doc-1"},
            formula_id="Equation 4",
        )
    ] == ["formula-1"]
    assert (
        store.lookup_asset_links(
            authorized_doc_ids={"doc-2"},
            page=2,
            figure_id="Figure 1",
            limit=0,
        )
        == []
    )


def test_visual_asset_link_records_relative_reference_hash_and_dimensions(
    tmp_path,
) -> None:
    from graph_rag.assets import build_visual_asset_links

    upload_root = tmp_path / "uploads"
    image_path = upload_root / "user-1" / "doc-1" / "page-2.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (13, 17), "white").save(image_path)
    expected_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    element = SimpleNamespace(
        id="visual-1",
        type="figure",
        page_number=2,
        image_path=str(image_path),
        bbox=[0, 0, 13, 17],
        summary="A source figure.",
        figure_reference="Figure 1",
    )

    links = build_visual_asset_links(
        doc_id="doc-1",
        elements=[element],
        upload_root=upload_root,
    )

    assert links[0].storage_reference == "user-1/doc-1/page-2.png"
    assert links[0].sha256 == expected_hash
    assert links[0].width == 13
    assert links[0].height == 17


@pytest.mark.parametrize(
    "storage_reference",
    [
        "C:/outside/secret.png",
        "C:outside/secret.png",
        "/outside/secret.png",
        "\\\\server\\share\\secret.png",
        "../doc-2/secret.png",
        "user-1/../doc-2/secret.png",
        "https://example.test/page.png",
        "file:///outside/secret.png",
        "data:image/png;base64,AAAA",
        "user-1\\doc-1\\page.png",
        "user-1//doc-1/page.png",
        "./user-1/doc-1/page.png",
    ],
)
def test_graph_asset_link_rejects_noncanonical_storage_references(
    storage_reference: str,
) -> None:
    with pytest.raises(ValidationError):
        GraphAssetLink(
            asset_id="invalid-reference",
            doc_id="doc-1",
            asset_type="figure",
            storage_reference=storage_reference,
        )


def test_graph_asset_link_accepts_legacy_none_and_canonical_relative_reference() -> (
    None
):
    legacy = GraphAssetLink(
        asset_id="legacy",
        doc_id="doc-1",
        asset_type="figure",
    )
    canonical = GraphAssetLink(
        asset_id="canonical",
        doc_id="doc-1",
        asset_type="figure",
        storage_reference="user-1/doc-1/page.png",
    )

    assert legacy.storage_reference is None
    assert canonical.storage_reference == "user-1/doc-1/page.png"


def test_store_revalidates_storage_reference_at_record_and_write_boundaries(
    tmp_path,
) -> None:
    invalid = GraphAssetLink.model_construct(
        asset_id="bypassed",
        doc_id="doc-1",
        asset_type="figure",
        storage_reference="data:image/png;base64,AAAA",
    )
    store = GraphStore("user-1", storage_dir=tmp_path)

    with pytest.raises(ValidationError):
        store.record_asset_link(invalid)

    store.asset_links[invalid.asset_id] = invalid
    with pytest.raises(ValidationError):
        store.save_sidecars()
