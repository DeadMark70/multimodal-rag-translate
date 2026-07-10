from __future__ import annotations

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
    from types import SimpleNamespace
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
