"""Deterministic parsing helpers for graph-to-document asset links."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

from graph_rag.schemas import GraphAssetLink

_PAGE_MARKER = re.compile(r"\[\[PAGE_(\d+)\]\]")
_CAPTION = re.compile(
    r"^\s*((?:table|fig(?:ure)?|圖|表)\s*[\d一二三四五六七八九十]+\b[^\n]*)",
    re.IGNORECASE,
)
_FORMULA = re.compile(r"\$\$[\s\S]+?\$\$|\\\\\[[\s\S]+?\\\\\]")


def _asset_id(*, doc_id: str, asset_type: str, page: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{doc_id}|{asset_type}|{page}|{text}".encode("utf-8")
    ).hexdigest()[:20]
    return f"asset:{asset_type}:{digest}"


def _asset_link(
    *,
    doc_id: str,
    page: int,
    asset_type: str,
    text: str,
    caption: str | None = None,
) -> GraphAssetLink:
    normalized = text.strip()
    return GraphAssetLink(
        asset_id=_asset_id(
            doc_id=doc_id,
            asset_type=asset_type,
            page=page,
            text=normalized,
        ),
        doc_id=doc_id,
        page=page,
        asset_type=asset_type,
        caption=caption,
        text_or_markdown=normalized,
        asset_text_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        asset_parse_status="parsed",
    )


def _table_links(doc_id: str, page: int, section: str) -> list[GraphAssetLink]:
    links: list[GraphAssetLink] = []
    lines = section.splitlines()
    index = 0
    while index + 1 < len(lines):
        header = lines[index].strip()
        separator = lines[index + 1].strip()
        if "|" not in header or not re.fullmatch(r"\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?", separator):
            index += 1
            continue
        end = index + 2
        while end < len(lines) and "|" in lines[end]:
            end += 1
        table = "\n".join(lines[index:end]).strip()
        caption = None
        for previous in reversed(lines[:index]):
            candidate = previous.strip()
            if not candidate:
                continue
            match = _CAPTION.match(candidate)
            if match:
                caption = match.group(1)
            break
        links.append(
            _asset_link(
                doc_id=doc_id,
                page=page,
                asset_type="table",
                text=table,
                caption=caption,
            )
        )
        index = end
    return links


def extract_markdown_asset_links(
    *,
    doc_id: str,
    markdown_text: str,
) -> list[GraphAssetLink]:
    """Extract only explicit Markdown assets and page markers, without fabricating links."""
    links: list[GraphAssetLink] = []
    page = 1
    cursor = 0
    for marker in _PAGE_MARKER.finditer(markdown_text):
        section = markdown_text[cursor : marker.start()]
        links.extend(_links_for_section(doc_id=doc_id, page=page, section=section))
        page = int(marker.group(1))
        cursor = marker.end()
    links.extend(_links_for_section(doc_id=doc_id, page=page, section=markdown_text[cursor:]))
    return links


def _links_for_section(*, doc_id: str, page: int, section: str) -> list[GraphAssetLink]:
    links = _table_links(doc_id, page, section)
    for match in _FORMULA.finditer(section):
        links.append(
            _asset_link(
                doc_id=doc_id,
                page=page,
                asset_type="formula",
                text=match.group(0),
            )
        )
    for line in section.splitlines():
        match = _CAPTION.match(line)
        if match:
            caption = match.group(1)
            links.append(
                _asset_link(
                    doc_id=doc_id,
                    page=page,
                    asset_type="caption",
                    text=caption,
                    caption=caption,
                )
            )
    return links


def build_visual_asset_links(
    *,
    doc_id: str,
    elements: Sequence[Any],
) -> list[GraphAssetLink]:
    """Prepare links for valid visual summaries before their vector metadata is written."""
    links: list[GraphAssetLink] = []
    for element in elements:
        summary = str(getattr(element, "summary", "") or "").strip()
        if not summary or "error" in summary.lower() or "錯誤" in summary:
            continue
        raw_type = getattr(getattr(element, "type", None), "value", getattr(element, "type", ""))
        asset_type = str(raw_type).lower()
        if asset_type not in {"table", "figure", "formula"}:
            continue
        page = int(getattr(element, "page_number", 0) or 0)
        element_id = str(getattr(element, "id", "") or "")
        asset_id = _asset_id(
            doc_id=doc_id,
            asset_type=asset_type,
            page=page,
            text=f"{element_id}|{summary}",
        )
        try:
            setattr(element, "asset_id", asset_id)
        except (AttributeError, TypeError):
            pass
        bbox = getattr(element, "bbox", None)
        links.append(
            GraphAssetLink(
                asset_id=asset_id,
                doc_id=doc_id,
                page=page or None,
                asset_type=asset_type,
                caption=getattr(element, "figure_reference", None),
                text_or_markdown=summary,
                asset_text_hash=hashlib.sha256(summary.encode("utf-8")).hexdigest(),
                asset_parse_status="parsed",
                bbox=[float(value) for value in bbox] if bbox else None,
                source_chunk_id=f"graph:asset:{asset_id}",
            )
        )
    return links
