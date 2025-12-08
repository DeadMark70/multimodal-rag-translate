"""
Proposition Chunker

Decomposes complex sentences into atomic propositions for more precise retrieval.
Each proposition is a self-contained, factual statement that can be independently indexed.
"""

# Standard library
import asyncio
import logging
import re
from typing import List, Optional

# Third-party
from langchain.schema import Document
from langchain_core.messages import HumanMessage

# Local application
from core.llm_factory import get_llm

# Configure logging
logger = logging.getLogger(__name__)

# Prompt for proposition extraction
_PROPOSITION_PROMPT = """你是一個文本分析專家。請將下方的文本分解為多個「原子命題」(Atomic Propositions)。

原子命題的定義：
1. 每個命題只包含一個事實或論斷
2. 命題是自洽的，不依賴上下文就能理解
3. 代名詞（如「它」、「他」）應替換為具體名稱
4. 保留原文的關鍵資訊，不要添加推論

範例輸入：
"Tesla 在 2023 年第三季的營收增長了 20%，同時其在歐洲的市場份額有所下降。"

範例輸出：
1. Tesla 在 2023 年第三季的營收增長了 20%。
2. Tesla 在歐洲的市場份額有所下降。

---

請分解以下文本，每行輸出一個命題（用數字編號）：

{text}

命題列表："""


class PropositionChunker:
    """
    Decomposes documents into atomic propositions using LLM.
    
    Useful for improving retrieval precision when documents contain
    compound statements or multiple facts per sentence.
    """
    
    def __init__(self, max_concurrent: int = 3) -> None:
        """
        Initializes the proposition chunker.
        
        Args:
            max_concurrent: Maximum concurrent LLM API calls.
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def extract_propositions(
        self,
        document: Document,
        min_text_length: int = 50,
    ) -> List[Document]:
        """
        Extracts atomic propositions from a document.
        
        Args:
            document: Source document to decompose.
            min_text_length: Minimum text length to process (skip short texts).
            
        Returns:
            List of Documents, each containing one atomic proposition.
        """
        text = document.page_content.strip()
        
        # Skip very short texts
        if len(text) < min_text_length:
            return [document]
        
        async with self._semaphore:
            try:
                llm = get_llm("proposition_extraction")
                
                prompt = _PROPOSITION_PROMPT.format(text=text[:2000])  # Limit input
                message = HumanMessage(content=prompt)
                
                response = await llm.ainvoke([message])
                propositions = self._parse_propositions(response.content)
                
                if not propositions:
                    # If extraction failed, return original
                    return [document]
                
                # Create proposition documents
                prop_docs = []
                for i, prop in enumerate(propositions):
                    prop_metadata = document.metadata.copy()
                    prop_metadata["is_proposition"] = True
                    prop_metadata["proposition_index"] = i
                    prop_metadata["parent_chunk_id"] = document.metadata.get(
                        "unique_chunk_id", ""
                    )
                    
                    prop_docs.append(Document(
                        page_content=prop,
                        metadata=prop_metadata
                    ))
                
                logger.debug(f"Extracted {len(prop_docs)} propositions from chunk")
                return prop_docs
                
            except Exception as e:
                logger.warning(f"Proposition extraction failed: {e}")
                return [document]
    
    def _parse_propositions(self, response: str) -> List[str]:
        """
        Parses LLM response into list of propositions.
        
        Args:
            response: Raw LLM response text.
            
        Returns:
            List of proposition strings.
        """
        propositions = []
        
        # Match numbered items (1. xxx, 2. xxx, etc.)
        pattern = r'^\d+[\.\)]\s*(.+)$'
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                prop = match.group(1).strip()
                if prop and len(prop) > 10:  # Filter out very short props
                    propositions.append(prop)
            elif line and not line.startswith('#') and len(line) > 20:
                # Non-numbered lines that look like valid propositions
                propositions.append(line)
        
        return propositions
    
    async def extract_propositions_batch(
        self,
        documents: List[Document],
        min_text_length: int = 50,
    ) -> List[Document]:
        """
        Extracts propositions from multiple documents concurrently.
        
        Args:
            documents: List of source documents.
            min_text_length: Minimum text length to process.
            
        Returns:
            Flattened list of proposition documents.
        """
        if not documents:
            return documents
        
        logger.info(f"Extracting propositions from {len(documents)} documents...")
        
        tasks = [
            self.extract_propositions(doc, min_text_length)
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_props = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Document {i} proposition extraction failed: {result}")
                all_props.append(documents[i])  # Use original
            else:
                all_props.extend(result)
        
        logger.info(f"Extracted {len(all_props)} propositions from {len(documents)} sources")
        return all_props


async def extract_propositions_from_documents(
    documents: List[Document],
    max_concurrent: int = 3,
    enabled: bool = True,
    min_text_length: int = 50,
) -> List[Document]:
    """
    Convenience function to extract propositions from documents.
    
    Args:
        documents: List of documents to process.
        max_concurrent: Maximum concurrent LLM calls.
        enabled: If False, returns documents unchanged.
        min_text_length: Minimum text length to process.
        
    Returns:
        List of proposition documents.
    """
    if not enabled:
        logger.debug("Proposition extraction disabled")
        return documents
    
    chunker = PropositionChunker(max_concurrent=max_concurrent)
    return await chunker.extract_propositions_batch(documents, min_text_length)
