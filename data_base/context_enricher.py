"""
Context Enricher for RAG Enhancement

Enriches text chunks with contextual prefixes using LLM to help with
coreference resolution and improve retrieval accuracy.
"""

# Standard library
import asyncio
import logging
from typing import List

# Third-party
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Local application
from core.providers import get_llm
from core.prompt_loader import format_rag_pipeline_prompt

# Configure logging
logger = logging.getLogger(__name__)


class ContextEnricher:
    """
    Enriches text chunks with contextual prefixes using LLM.
    
    This helps with:
    - Coreference resolution (pronouns → entities)
    - Topic clarification
    - Improving retrieval accuracy for ambiguous chunks
    
    Attributes:
        _semaphore: Rate limiter for LLM API calls.
    """
    
    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initializes the context enricher.
        
        Args:
            max_concurrent: Maximum concurrent LLM API calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def enrich_chunk(
        self,
        chunk: Document,
        document_title: str,
    ) -> Document:
        """
        Enriches a single chunk with contextual prefix.
        
        Args:
            chunk: Document chunk to enrich.
            document_title: Title of the source document.
            
        Returns:
            New Document with contextual prefix added to content.
        """
        async with self._semaphore:
            try:
                llm = get_llm("context_generation")
                
                prompt = format_rag_pipeline_prompt(
                    "context_enrichment",
                    document_title=document_title,
                    chunk_content=chunk.page_content[:1000],  # Limit input size
                )
                
                message = HumanMessage(content=prompt)
                response = await llm.ainvoke([message])
                
                context_prefix = response.content.strip()
                
                # Create enriched content
                enriched_content = f"<context>{context_prefix}</context> {chunk.page_content}"
                
                # Create new document with enriched content
                enriched_metadata = chunk.metadata.copy()
                enriched_metadata["has_context_prefix"] = True
                enriched_metadata["context_prefix"] = context_prefix
                
                return Document(
                    page_content=enriched_content,
                    metadata=enriched_metadata
                )
                
            except Exception as e:
                logger.warning(f"Context enrichment failed for chunk: {e}")
                # Return original chunk if enrichment fails
                return chunk
    
    async def enrich_chunks(
        self,
        chunks: List[Document],
        document_title: str,
        skip_if_short: int = 50,
    ) -> List[Document]:
        """
        Enriches multiple chunks with contextual prefixes.
        
        Args:
            chunks: List of document chunks to enrich.
            document_title: Title of the source document.
            skip_if_short: Skip context generation for chunks shorter than this.
            
        Returns:
            List of enriched documents.
        """
        if not chunks:
            return chunks
        
        logger.info(f"Enriching {len(chunks)} chunks with context...")
        
        tasks = []
        for chunk in chunks:
            # Skip very short chunks
            if len(chunk.page_content) < skip_if_short:
                tasks.append(asyncio.create_task(self._return_as_is(chunk)))
            else:
                tasks.append(
                    asyncio.create_task(self.enrich_chunk(chunk, document_title))
                )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        enriched = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Chunk {i} enrichment failed: {result}")
                enriched.append(chunks[i])  # Use original
            else:
                enriched.append(result)
        
        successful = sum(1 for d in enriched if d.metadata.get("has_context_prefix"))
        logger.info(f"Context enrichment complete: {successful}/{len(enriched)} chunks enriched")
        
        return enriched
    
    async def _return_as_is(self, chunk: Document) -> Document:
        """Helper to return chunk unchanged in async context."""
        return chunk


async def enrich_documents_with_context(
    documents: List[Document],
    document_title: str,
    max_concurrent: int = 3,
    enabled: bool = True,
) -> List[Document]:
    """
    Convenience function to enrich documents with contextual prefixes.
    
    Args:
        documents: List of documents to enrich.
        document_title: Title of the source document.
        max_concurrent: Maximum concurrent LLM calls.
        enabled: If False, returns documents unchanged.
        
    Returns:
        List of enriched documents.
    """
    if not enabled:
        logger.debug("Context enrichment disabled, returning original documents")
        return documents
    
    enricher = ContextEnricher(max_concurrent=max_concurrent)
    return await enricher.enrich_chunks(documents, document_title)
