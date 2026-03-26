from data_base.document_metadata import (
    CANONICAL_DOC_ID_KEY,
    LEGACY_DOC_ID_KEY,
    get_document_id,
    matches_document_id,
    with_document_id,
)


def test_with_document_id_writes_canonical_key_only() -> None:
    metadata = with_document_id(
        {"page_number": 1, LEGACY_DOC_ID_KEY: "legacy-doc"},
        "doc-123",
    )

    assert metadata[CANONICAL_DOC_ID_KEY] == "doc-123"
    assert LEGACY_DOC_ID_KEY not in metadata


def test_get_document_id_falls_back_to_legacy_key() -> None:
    metadata = {LEGACY_DOC_ID_KEY: "legacy-doc"}
    assert get_document_id(metadata) == "legacy-doc"


def test_matches_document_id_supports_canonical_and_legacy_keys() -> None:
    assert matches_document_id({CANONICAL_DOC_ID_KEY: "doc-1"}, "doc-1") is True
    assert matches_document_id({LEGACY_DOC_ID_KEY: "doc-2"}, "doc-2") is True
    assert matches_document_id({CANONICAL_DOC_ID_KEY: "doc-1"}, "doc-2") is False
