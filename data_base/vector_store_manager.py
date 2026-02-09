"""
Vector Store Manager

Manages per-user FAISS vector stores for RAG indexing and retrieval.
Supports both traditional and semantic chunking methods.

Uses Google Gemini Embedding API for vector generation.
"""

# Standard library
import logging
import os
import pickle
import time
from typing import List, Optional, Literal

# Third-party
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Local application
from data_base.word_chunk_strategy import split_markdown
from multimodal_rag.schemas import ExtractedDocument

# Configure logging
logger = logging.getLogger(__name__)

# Embedding model (shared globally - API based, no local model to load)
global_embeddings_model: Optional[GoogleGenerativeAIEmbeddings] = None

# User RAG files are stored here
BASE_UPLOAD_FOLDER = "uploads"

# Chunking method type
ChunkingMethod = Literal["recursive", "semantic"]


def get_user_vector_store_path(user_id: str) -> str:
    """
    Returns the path to user's RAG index folder.

    Args:
        user_id: User's ID.

    Returns:
        Path to user's rag_index folder.
    """
    return os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id, "rag_index"))


async def initialize_embeddings(embedding_model_name: str = "models/gemini-embedding-001") -> None:
    """
    Initializes the Google Embedding model via API.

    This should be called once during application startup.

    Args:
        embedding_model_name: Google Embedding model name.

    Raises:
        RuntimeError: If API key is not set or initialization fails.
    """
    global global_embeddings_model

    if global_embeddings_model is not None:
        logger.info("Embedding model already initialized")
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set for embedding model")

    logger.info(f"Initializing Google Embedding: {embedding_model_name}")

    try:
        global_embeddings_model = GoogleGenerativeAIEmbeddings(
            model=embedding_model_name,
            google_api_key=api_key
        )
        logger.info("Google Embedding model ready (API mode)")
    except (RuntimeError, OSError, ValueError) as e:
        logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
        raise RuntimeError(f"Embedding model initialization failed: {e}")


def get_embeddings() -> Optional[GoogleGenerativeAIEmbeddings]:
    """
    Returns the global embeddings model.

    This is useful for modules that need access to the embedding model
    without triggering circular imports.

    Returns:
        The global GoogleGenerativeAIEmbeddings instance, or None if not initialized.
    """
    return global_embeddings_model


def _add_documents_with_retry(
    vector_db: FAISS,
    chunks: List[Document],
    max_retries: int = 3,
    base_delay: float = 30.0,
) -> None:
    """
    Adds documents to FAISS with exponential backoff retry for rate limits.

    Handles 429 RESOURCE_EXHAUSTED errors from Gemini Embedding API
    by waiting and retrying with exponential backoff.

    Args:
        vector_db: FAISS vector store instance.
        chunks: List of documents to add.
        max_retries: Maximum retry attempts (default 3).
        base_delay: Initial delay in seconds (default 30).

    Raises:
        RuntimeError: If all retries exhausted.
    """
    from langchain_google_genai._common import GoogleGenerativeAIError

    for attempt in range(max_retries + 1):
        try:
            vector_db.add_documents(chunks)
            if attempt > 0:
                logger.info(f"Embedding succeeded on retry attempt {attempt}")
            return
        except GoogleGenerativeAIError as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # 30, 60, 120 seconds
                    logger.warning(
                        f"Embedding rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.0f}s before retry..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Embedding failed after {max_retries + 1} attempts due to rate limits"
                    )
                    raise RuntimeError(
                        f"Embedding rate limit exceeded after {max_retries + 1} attempts"
                    ) from e
            else:
                # Non-rate-limit error, re-raise immediately
                raise


def _create_faiss_with_retry(
    chunks: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings,
    max_retries: int = 3,
    base_delay: float = 30.0,
) -> FAISS:
    """
    Creates FAISS from documents with exponential backoff retry for rate limits.

    Args:
        chunks: List of documents to index.
        embeddings: Embedding model.
        max_retries: Maximum retry attempts.
        base_delay: Initial delay in seconds.

    Returns:
        FAISS vector store.

    Raises:
        RuntimeError: If all retries exhausted.
    """
    from langchain_google_genai._common import GoogleGenerativeAIError

    for attempt in range(max_retries + 1):
        try:
            vector_db = FAISS.from_documents(chunks, embeddings)
            if attempt > 0:
                logger.info(f"FAISS creation succeeded on retry attempt {attempt}")
            return vector_db
        except GoogleGenerativeAIError as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Embedding rate limit hit (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Waiting {delay:.0f}s before retry..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"FAISS creation failed after {max_retries + 1} attempts"
                    )
                    raise RuntimeError(
                        f"Embedding rate limit exceeded after {max_retries + 1} attempts"
                    ) from e
            else:
                raise

    # Should not reach here
    raise RuntimeError("Unexpected error in _create_faiss_with_retry")


def index_extracted_document(user_id: str, doc: ExtractedDocument) -> None:
    """
    Indexes an ExtractedDocument (from multimodal extraction) into FAISS.

    Indexes both text chunks and visual element summaries.

    Args:
        user_id: User's ID.
        doc: ExtractedDocument from structure analyzer.

    Raises:
        ValueError: If embedding model not initialized.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        raise ValueError("Embedding model not initialized")

    documents_to_add: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    # Step A1: Process text chunks
    for chunk in doc.text_chunks:
        if not chunk.content or not chunk.content.strip():
            continue

        splits = text_splitter.split_text(chunk.content)
        for i, split_text in enumerate(splits):
            if not split_text.strip():
                continue
            documents_to_add.append(Document(
                page_content=split_text,
                metadata={
                    "source": "text",
                    "doc_id": str(doc.doc_id),
                    "page": chunk.page_number,
                    "chunk_id": f"{chunk.chunk_id}_{i}"
                }
            ))

    logger.info(f"Prepared {len(documents_to_add)} text chunks for indexing")

    # Step A2: Process visual elements
    visual_count = 0
    for element in doc.visual_elements:
        if not element.summary or not element.summary.strip():
            continue
        if "Error" in element.summary:
            continue

        documents_to_add.append(Document(
            page_content=element.summary,
            metadata={
                "source": "image",
                "doc_id": str(doc.doc_id),
                "type": element.type.value,
                "page": element.page_number,
                "image_path": element.image_path,
                "bbox": str(element.bbox)
            }
        ))
        visual_count += 1

    logger.info(f"Prepared {visual_count} visual element summaries for indexing")

    if not documents_to_add:
        logger.warning("No documents to add, skipping indexing")
        return

    # Step B: Write to FAISS
    user_index_path = get_user_vector_store_path(user_id)
    os.makedirs(user_index_path, exist_ok=True)

    try:
        vector_db = None
        faiss_file = os.path.join(user_index_path, "index.faiss")

        if os.path.exists(faiss_file):
            logger.info(f"Loading existing index for user {user_id}...")
            vector_db = FAISS.load_local(
                user_index_path,
                global_embeddings_model,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            vector_db.add_documents(documents_to_add)
        else:
            logger.info(f"Creating new index for user {user_id}...")
            vector_db = FAISS.from_documents(documents_to_add, global_embeddings_model)

        vector_db.save_local(user_index_path, index_name="index")
        logger.info(f"Successfully indexed {len(documents_to_add)} documents")

    except (RuntimeError, OSError, pickle.PicklingError, ValueError) as e:
        logger.error(f"Indexing error: {e}", exc_info=True)


def add_visual_summaries_to_knowledge_base(
    user_id: str,
    doc_id: str,
    elements: List,
) -> int:
    """
    Adds visual element summaries to user's knowledge base.

    This function is called after ImageSummarizer has generated summaries
    for visual elements extracted from a document.

    Args:
        user_id: User's ID.
        doc_id: Document UUID.
        elements: List of VisualElement objects with summaries.

    Returns:
        Number of visual summaries indexed.

    Raises:
        ValueError: If embedding model not initialized.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        raise ValueError("Embedding model not initialized")

    # Filter elements with valid summaries
    documents_to_add: List[Document] = []
    for element in elements:
        if not element.summary or not element.summary.strip():
            continue
        if "Error" in element.summary or "錯誤" in element.summary:
            continue

        documents_to_add.append(Document(
            page_content=element.summary,
            metadata={
                "source": "image",
                "type": element.type.value if hasattr(element.type, 'value') else str(element.type),
                "doc_id": doc_id,
                "page": element.page_number,
                "image_path": element.image_path,
                "context": element.context_text or "",
                "figure_ref": element.figure_reference or "",
            }
        ))

    if not documents_to_add:
        logger.info("No visual summaries to index")
        return 0

    logger.info(f"Indexing {len(documents_to_add)} visual summaries for user {user_id}")

    # Add to existing FAISS index
    user_index_path = get_user_vector_store_path(user_id)
    os.makedirs(user_index_path, exist_ok=True)

    try:
        faiss_file = os.path.join(user_index_path, "index.faiss")

        if os.path.exists(faiss_file):
            # Append to existing index
            vector_db = FAISS.load_local(
                user_index_path,
                global_embeddings_model,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            vector_db.add_documents(documents_to_add)
        else:
            # Create new index
            logger.info(f"Creating new index for user {user_id}")
            vector_db = FAISS.from_documents(documents_to_add, global_embeddings_model)

        vector_db.save_local(user_index_path, index_name="index")
        logger.info(f"Successfully indexed {len(documents_to_add)} visual summaries")
        return len(documents_to_add)

    except (RuntimeError, OSError, pickle.PicklingError, ValueError) as e:
        logger.error(f"Visual summary indexing error: {e}", exc_info=True)
        return 0

def get_user_retriever(user_id: str, k: int = 3):
    """
    Gets a hybrid retriever for a specific user.

    Combines Vector Search (FAISS) with Keyword Search (BM25).

    Args:
        user_id: User's ID.
        k: Number of documents to retrieve.

    Returns:
        EnsembleRetriever or None if index doesn't exist.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        logger.warning("Embedding model not initialized")
        return None

    user_index_path = get_user_vector_store_path(user_id)
    faiss_file = os.path.join(user_index_path, "index.faiss")

    if not os.path.exists(faiss_file):
        return None

    try:
        # 1. Load FAISS (vector search)
        vector_db = FAISS.load_local(
            user_index_path,
            global_embeddings_model,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        faiss_retriever = vector_db.as_retriever(search_kwargs={"k": k})

        # 2. Build BM25 (keyword search)
        documents = list(vector_db.docstore._dict.values())
        if not documents:
            return faiss_retriever

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k

        # 3. Build ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )

        logger.debug(f"Loaded hybrid retriever for user {user_id}")
        return ensemble_retriever

    except (RuntimeError, OSError, pickle.UnpicklingError) as e:
        logger.error(f"Error loading user {user_id} index: {e}", exc_info=True)
        return None


async def add_markdown_to_knowledge_base(
    user_id: str,
    markdown_text: str,
    pdf_title: str,
    doc_id: str,
    k_retriever: int = 3,
    chunking_method: ChunkingMethod = "recursive",
    enable_context_enrichment: bool = False,
):
    """
    Adds a document to a user's knowledge base.

    Supports both traditional recursive chunking and semantic chunking.

    Args:
        user_id: User's ID.
        markdown_text: Markdown content to index.
        pdf_title: Title of the source PDF.
        doc_id: Document ID.
        k_retriever: Number of documents for retriever.
        chunking_method: "recursive" (default) or "semantic".
        enable_context_enrichment: If True, enrich chunks with LLM-generated context.

    Returns:
        Retriever for the updated index.

    Raises:
        RuntimeError: If embedding model not initialized.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        raise RuntimeError("Embedding model not initialized")

    # 1. Split document using the specified method
    if chunking_method == "semantic":
        logger.info(f"User {user_id}: Using semantic chunking")
        chunks = await split_markdown(
            markdown_text,
            pdf_title,
            doc_id,
            chunking_method="semantic",
            embeddings=global_embeddings_model,
        )
    else:
        chunks = await split_markdown(
            markdown_text,
            pdf_title,
            doc_id,
            chunk_size=800,
            overlap=150,
            chunking_method="recursive",
        )

    logger.info(f"User {user_id}: Document split into {len(chunks)} chunks using {chunking_method}")

    # 2. Optional: Enrich chunks with context
    if enable_context_enrichment and chunks:
        try:
            from data_base.context_enricher import enrich_documents_with_context
            chunks = await enrich_documents_with_context(
                documents=chunks,
                document_title=pdf_title,
                max_concurrent=3,
                enabled=True,
            )
            logger.info(f"User {user_id}: Context enrichment completed")
        except ImportError:
            logger.warning("Context enricher not available, skipping enrichment")
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Context enrichment failed (non-fatal): {e}")

    # 2. Prepare path
    user_index_path = get_user_vector_store_path(user_id)
    os.makedirs(user_index_path, exist_ok=True)

    # 3. Load existing or create new index
    vector_db = None
    faiss_file = os.path.join(user_index_path, "index.faiss")

    if os.path.exists(faiss_file):
        try:
            logger.info(f"Loading existing index for user {user_id}...")
            vector_db = FAISS.load_local(
                user_index_path,
                global_embeddings_model,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            _add_documents_with_retry(vector_db, chunks)
        except (RuntimeError, OSError, pickle.UnpicklingError) as e:
            logger.warning(f"Error loading existing index: {e}, creating new one")
            vector_db = None

    if vector_db is None:
        logger.info(f"Creating new index for user {user_id}...")
        vector_db = _create_faiss_with_retry(chunks, global_embeddings_model)

    # 4. Save to disk
    try:
        vector_db.save_local(user_index_path, index_name="index")
        logger.info(f"Index saved to {user_index_path}")
    except (OSError, IOError) as e:
        logger.warning(f"Could not save FAISS index: {e}")

    # 5. Return updated retriever
    return vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k_retriever})


def delete_document_from_knowledge_base(user_id: str, doc_id: str) -> bool:
    """
    Deletes all vectors for a specific document from user's FAISS index.

    Also cleans up parent document store if using hierarchical indexing.

    Args:
        user_id: User's ID.
        doc_id: Document ID to delete.

    Returns:
        True if deletion successful, False otherwise.
    """
    global global_embeddings_model

    user_index_path = get_user_vector_store_path(user_id)
    faiss_file = os.path.join(user_index_path, "index.faiss")

    if not os.path.exists(faiss_file):
        logger.info(f"Index not found for user {user_id}, skipping")
        return False

    try:
        # 1. Load index
        vector_db = FAISS.load_local(
            user_index_path,
            global_embeddings_model,
            index_name="index",
            allow_dangerous_deserialization=True
        )

        # 2. Find all vector IDs for this doc_id
        ids_to_delete = []
        for key, doc in vector_db.docstore._dict.items():
            # Check both field names for compatibility
            if (doc.metadata.get("original_doc_uid") == doc_id or 
                doc.metadata.get("doc_id") == doc_id):
                ids_to_delete.append(key)

        # 3. Execute deletion
        if ids_to_delete:
            logger.info(f"Deleting {len(ids_to_delete)} chunks for doc {doc_id}")
            vector_db.delete(ids_to_delete)
            vector_db.save_local(user_index_path, index_name="index")
            logger.info("Index updated and saved")

        # 4. Clean up parent store if exists
        try:
            from data_base.parent_child_store import ParentDocumentStore
            parent_store = ParentDocumentStore(user_id)
            deleted_parents = parent_store.delete_by_doc_id(doc_id)
            if deleted_parents > 0:
                logger.info(f"Deleted {deleted_parents} parent documents")
        except ImportError:
            pass  # Parent store not available
        except (IOError, KeyError, RuntimeError) as e:
            logger.warning(f"Parent store cleanup failed (non-fatal): {e}")

        return len(ids_to_delete) > 0

    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        return False


def get_user_parent_child_retriever(
    user_id: str,
    k: int = 6,
    return_parents: bool = True,
):
    """
    Gets a retriever that uses parent-child indexing.

    Retrieves child chunks for precision, but returns parent chunks for context.

    Args:
        user_id: User's ID.
        k: Number of child documents to retrieve.
        return_parents: If True, returns parent documents instead of children.

    Returns:
        A callable retriever function, or None if index doesn't exist.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        logger.warning("Embedding model not initialized")
        return None

    user_index_path = get_user_vector_store_path(user_id)
    faiss_file = os.path.join(user_index_path, "index.faiss")

    if not os.path.exists(faiss_file):
        return None

    try:
        # Load FAISS
        vector_db = FAISS.load_local(
            user_index_path,
            global_embeddings_model,
            index_name="index",
            allow_dangerous_deserialization=True
        )
        base_retriever = vector_db.as_retriever(search_kwargs={"k": k})

        if not return_parents:
            return base_retriever

        # Create wrapper that fetches parents
        from data_base.parent_child_store import ParentDocumentStore
        parent_store = ParentDocumentStore(user_id)

        class ParentChildRetriever:
            """Retriever that returns parent documents for child matches."""

            def __init__(self, child_retriever, parent_store):
                self._child_retriever = child_retriever
                self._parent_store = parent_store

            def invoke(self, query: str) -> List[Document]:
                """Retrieves parent documents for matching children."""
                children = self._child_retriever.invoke(query)
                return self._parent_store.get_parents_for_children(children)

            async def ainvoke(self, query: str) -> List[Document]:
                """Async version of invoke."""
                # FAISS retriever invoke is sync, so we just call it
                return self.invoke(query)

        return ParentChildRetriever(base_retriever, parent_store)

    except ImportError:
        logger.warning("Parent store not available, using standard retriever")
        return vector_db.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        logger.error(f"Error creating parent-child retriever: {e}", exc_info=True)
        return None


async def add_markdown_with_hierarchical_indexing(
    user_id: str,
    markdown_text: str,
    pdf_title: str,
    doc_id: str,
    parent_chunk_size: int = 2000,
    child_chunk_size: int = 400,
    enable_proposition_indexing: bool = False,
):
    """
    Adds a document with parent-child hierarchical indexing.

    Creates larger parent chunks for context and smaller child chunks
    for precise matching. Optionally decomposes into atomic propositions.

    Args:
        user_id: User's ID.
        markdown_text: Markdown content to index.
        pdf_title: Title of the source PDF.
        doc_id: Document ID.
        parent_chunk_size: Target size for parent chunks.
        child_chunk_size: Target size for child chunks.
        enable_proposition_indexing: If True, decompose into atomic propositions.

    Returns:
        Number of child chunks indexed.

    Raises:
        RuntimeError: If embedding model not initialized.
    """
    global global_embeddings_model

    if global_embeddings_model is None:
        raise RuntimeError("Embedding model not initialized")

    # 1. Initial chunking
    from data_base.word_chunk_strategy import split_markdown
    initial_chunks = await split_markdown(
        markdown_text,
        pdf_title,
        doc_id,
        chunk_size=parent_chunk_size,
        overlap=200,
    )
    logger.info(f"User {user_id}: Initial split into {len(initial_chunks)} chunks")

    # 2. Create parent-child hierarchy
    from data_base.parent_child_store import (
        ParentDocumentStore,
        create_parent_child_chunks,
    )

    parents, children = create_parent_child_chunks(
        initial_chunks,
        parent_chunk_size=parent_chunk_size,
        child_chunk_size=child_chunk_size,
    )

    # 3. Store parents
    parent_store = ParentDocumentStore(user_id)
    parent_store.add_parents(parents)

    # 4. Optional: Proposition indexing on children
    if enable_proposition_indexing:
        try:
            from data_base.proposition_chunker import extract_propositions_from_documents
            children = await extract_propositions_from_documents(
                children,
                max_concurrent=3,
                enabled=True,
            )
            logger.info(f"User {user_id}: Expanded to {len(children)} propositions")
        except ImportError:
            logger.warning("Proposition chunker not available")
        except Exception as e:
            logger.warning(f"Proposition extraction failed (non-fatal): {e}")

    # 5. Index children to FAISS
    if not children:
        logger.warning("No children to index")
        return 0

    user_index_path = get_user_vector_store_path(user_id)
    os.makedirs(user_index_path, exist_ok=True)

    vector_db = None
    faiss_file = os.path.join(user_index_path, "index.faiss")

    if os.path.exists(faiss_file):
        try:
            vector_db = FAISS.load_local(
                user_index_path,
                global_embeddings_model,
                index_name="index",
                allow_dangerous_deserialization=True
            )
            vector_db.add_documents(children)
        except (RuntimeError, OSError, pickle.UnpicklingError) as e:
            logger.warning(f"Error loading existing index: {e}")
            vector_db = None

    if vector_db is None:
        vector_db = FAISS.from_documents(children, global_embeddings_model)

    try:
        vector_db.save_local(user_index_path, index_name="index")
        logger.info(f"Indexed {len(children)} child chunks with hierarchical structure")
    except (OSError, IOError) as e:
        logger.warning(f"Could not save FAISS index: {e}")

    return len(children)
