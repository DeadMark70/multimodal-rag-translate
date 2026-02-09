"""
Markdown Chunking Strategy

Provides functions for splitting Markdown text into chunks for RAG indexing.
Supports both traditional recursive splitting and semantic-aware chunking.
"""

# Standard library
import logging
import re
from typing import List, Literal

# Third-party
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local application
from data_base.semantic_chunker import SemanticTextChunker

# Configure logging
logger = logging.getLogger(__name__)

# Chunking method type
ChunkingMethod = Literal["recursive", "semantic"]


async def split_markdown(
    markdown_text: str,
    pdf_title: str,
    original_doc_uid: str,
    chunk_size: int = 1000,
    overlap: int = 150,
    chunking_method: ChunkingMethod = "recursive",
    embeddings = None,  # Any embedding model with embed_documents method
) -> List[Document]:
    """
    Splits Markdown text into chunks with metadata.

    Supports two chunking methods:
    - "recursive": Traditional character-based splitting (default, backward compatible)
    - "semantic": Embedding-based semantic boundary detection

    Recognizes [[PAGE_N]] markers and preserves page information in chunk metadata.

    Args:
        markdown_text: Full Markdown text with page markers.
        pdf_title: Title of the source PDF.
        original_doc_uid: UUID of the source document.
        chunk_size: Maximum characters per chunk (for recursive method).
        overlap: Character overlap between chunks (for recursive method).
        chunking_method: "recursive" or "semantic".
        embeddings: HuggingFace embeddings (required for semantic chunking).

    Returns:
        List of LangChain Documents with metadata.

    Raises:
        ValueError: If semantic chunking requested but embeddings not provided.
    """
    # Validate semantic chunking requirements
    if chunking_method == "semantic" and embeddings is None:
        raise ValueError("embeddings parameter required for semantic chunking")

    all_chunks: List[Document] = []

    # Find page markers
    page_markers = [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"\[\[PAGE_\d+\]\]", markdown_text)
    ]

    # Process based on page markers
    if not page_markers:
        logger.debug("No page markers found, treating as single page")
        all_chunks = _process_page_content(
            page_content=markdown_text,
            page_number=1,
            pdf_title=pdf_title,
            original_doc_uid=original_doc_uid,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            overlap=overlap,
            embeddings=embeddings,
        )
    else:
        # Process each page
        for i in range(len(page_markers)):
            start_index = page_markers[i][1]
            current_page_number_str = page_markers[i][2].replace("[[PAGE_", "").replace("]]", "")
            current_page_number = int(current_page_number_str) if current_page_number_str.isdigit() else i + 1

            end_index = page_markers[i + 1][0] if i + 1 < len(page_markers) else len(markdown_text)
            page_content = markdown_text[start_index:end_index].strip()

            if page_content:
                page_chunks = _process_page_content(
                    page_content=page_content,
                    page_number=current_page_number,
                    pdf_title=pdf_title,
                    original_doc_uid=original_doc_uid,
                    chunking_method=chunking_method,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    embeddings=embeddings,
                )
                all_chunks.extend(page_chunks)

    logger.debug(f"Split into {len(all_chunks)} total chunks using {chunking_method} method")
    return all_chunks


def _process_page_content(
    page_content: str,
    page_number: int,
    pdf_title: str,
    original_doc_uid: str,
    chunking_method: ChunkingMethod,
    chunk_size: int,
    overlap: int,
    embeddings,  # Embedding model (any with embed_documents method)
) -> List[Document]:
    """
    Processes a single page's content into chunks.

    Args:
        page_content: Text content of the page.
        page_number: Page number for metadata.
        pdf_title: Title of the source PDF.
        original_doc_uid: UUID of the source document.
        chunking_method: Chunking method to use.
        chunk_size: Maximum chunk size (for recursive).
        overlap: Chunk overlap (for recursive).
        embeddings: Embeddings model (for semantic).

    Returns:
        List of Document chunks for this page.
    """
    base_metadata = {
        "book_title": pdf_title,
        "original_doc_uid": original_doc_uid,
        "page_number": page_number,
    }

    if chunking_method == "semantic" and embeddings is not None:
        # Use semantic chunking
        page_chunks = _split_with_semantic_chunker(
            text=page_content,
            metadata=base_metadata,
            embeddings=embeddings,
        )
    else:
        # Use recursive character splitting (default)
        doc_for_splitting = Document(
            page_content=page_content,
            metadata=base_metadata,
        )
        page_chunks = _split_single_document_recursively(
            doc_for_splitting, chunk_size, overlap
        )

    # Add chunk identifiers
    for i, chunk in enumerate(page_chunks):
        chunk.metadata["chunk_index_in_page"] = i
        chunk.metadata["unique_chunk_id"] = f"{original_doc_uid}_page_{page_number}_chunk_{i}"

    return page_chunks


def _split_with_semantic_chunker(
    text: str,
    metadata: dict,
    embeddings,  # Embedding model
) -> List[Document]:
    """
    Splits text using semantic boundary detection.

    Args:
        text: Text to split.
        metadata: Base metadata for chunks.
        embeddings: HuggingFace embeddings model.

    Returns:
        List of semantically coherent Document chunks.
    """
    try:
        chunker = SemanticTextChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90.0,
            min_chunk_size=100,
            max_chunk_size=2000,
        )
        return chunker.split_text(text, metadata)
    except RuntimeError as e:
        logger.warning(f"Semantic chunking failed, falling back to recursive: {e}")
        # Fallback to recursive
        doc = Document(page_content=text, metadata=metadata)
        return _split_single_document_recursively(doc, 1000, 150)


def _split_single_document_recursively(
    document: Document,
    chunk_size: int,
    overlap: int
) -> List[Document]:
    """
    Splits a single document using RecursiveCharacterTextSplitter.

    Args:
        document: LangChain Document to split.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between chunks.

    Returns:
        List of split Document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents([document])


async def split_markdown_semantic(
    markdown_text: str,
    pdf_title: str,
    original_doc_uid: str,
    embeddings,  # Embedding model
) -> List[Document]:
    """
    Convenience function for semantic chunking.

    Wrapper around split_markdown with semantic method preset.

    Args:
        markdown_text: Full Markdown text.
        pdf_title: Title of the source PDF.
        original_doc_uid: UUID of the source document.
        embeddings: HuggingFace embeddings model.

    Returns:
        List of semantically chunked Documents.
    """
    return await split_markdown(
        markdown_text=markdown_text,
        pdf_title=pdf_title,
        original_doc_uid=original_doc_uid,
        chunking_method="semantic",
        embeddings=embeddings,
    )

