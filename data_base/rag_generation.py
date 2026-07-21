"""Legacy-only answer generation for the v8 RAG compatibility wrapper.

This module deliberately owns multimodal prompt construction and the visual
verification synthesis loop.  Newer pipelines consume retrieval evidence
directly and must not import this legacy behavior.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from fastapi.concurrency import run_in_threadpool
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from core.llm_factory import get_llm_usage_metrics
from core.llm_usage_context import llm_accounting_phase
from core.prompt_loader import format_prompt
from data_base.document_metadata import get_document_id
from data_base.rag_pipeline_schemas import GeneratedRagAnswer
from data_base.repository import fetch_document_filenames

logger = logging.getLogger(__name__)

MAX_VISUAL_ITERATIONS = 2
ProgressCallback = Callable[[str, Optional[dict[str, Any]]], Awaitable[None]]


def _encode_image(image_path: str) -> Optional[str]:
    """Read an image as base64 while retaining legacy error handling."""
    image_path = os.path.normpath(image_path)
    if not os.path.exists(image_path):
        logger.warning("Image not found: %s", image_path)
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except IOError as error:
        logger.error("Error reading image %s: %s", image_path, error)
        return None


def parse_legacy_visual_tool_request(response: str) -> Optional[dict[str, str]]:
    """Parse the v8 ``VERIFY_IMAGE`` request format without widening it."""
    patterns = [
        r'\{\s*"action"\s*:\s*"VERIFY_IMAGE"\s*,\s*"path"\s*:\s*"([^"]+)"\s*,\s*"question"\s*:\s*"([^"]+)"\s*\}',
        r'\{\s*"action"\s*:\s*"VERIFY_IMAGE"[^}]*"path"\s*:\s*"([^"]+)"[^}]*"question"\s*:\s*"([^"]+)"[^}]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            return {"action": "VERIFY_IMAGE", "path": match.group(1), "question": match.group(2)}

    json_match = re.search(r"```(?:json)?\s*(\{[^`]+\})\s*```", response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if data.get("action") == "VERIFY_IMAGE" and data.get("path") and data.get("question"):
                return data
        except json.JSONDecodeError:
            pass
    return None


async def _execute_legacy_visual_verification_loop(
    *,
    initial_response: str,
    context: str,
    question: str,
    user_id: str,
    llm: Any,
    image_paths: list[str],
    force_once_if_not_triggered: bool,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Run the legacy image verification-and-resynthesis loop only for v8."""
    response = initial_response
    iteration = 0
    tool_results: list[dict[str, Any]] = []
    attempted = False
    forced_fallback_used = False
    force_pending = bool(force_once_if_not_triggered and image_paths)

    while iteration < MAX_VISUAL_ITERATIONS:
        tool_request = parse_legacy_visual_tool_request(response)
        tool_request_is_forced = False
        if not tool_request and force_pending:
            tool_request = {"action": "VERIFY_IMAGE", "path": image_paths[0], "question": question}
            tool_request_is_forced = True
            forced_fallback_used = True
            force_pending = False
            logger.info("Visual verification forced fallback triggered for image-aware route")
        elif tool_request:
            force_pending = False
        if not tool_request:
            break

        iteration += 1
        attempted = True
        from data_base.visual_tools import verify_image_details

        result = await verify_image_details(
            image_path=tool_request.get("path", ""),
            question=tool_request.get("question", ""),
            user_id=user_id,
        )
        tool_results.append(
            {
                "action": "VERIFY_IMAGE",
                "path": tool_request.get("path"),
                "question": tool_request.get("question"),
                "success": result.get("success"),
                "result": result.get("result") if result["success"] else result.get("error"),
                "forced_once": tool_request_is_forced,
            }
        )
        synthesis_prompt = format_prompt(
            "visual_verification_synthesis",
            context=context,
            question=question,
            initial_response=initial_response,
            verification_results=json.dumps(tool_results, ensure_ascii=False, indent=2),
        )
        with llm_accounting_phase("visual_verification"):
            response = (await llm.ainvoke([HumanMessage(content=synthesis_prompt)])).content
        if tool_request_is_forced:
            break

    return response, tool_results, {
        "visual_verification_attempted": attempted,
        "visual_tool_call_count": len(tool_results),
        "visual_force_fallback_used": forced_fallback_used,
    }


def legacy_source_doc_ids(documents: list[Document]) -> list[str]:
    """Project retrieved documents to the legacy source-ID response field."""
    return list({doc_id for document in documents if (doc_id := get_document_id(document.metadata))})


async def _emit_progress(
    progress_callback: Optional[ProgressCallback], stage: str, details: dict[str, Any]
) -> None:
    if progress_callback is not None:
        await progress_callback(stage, details)


async def generate_legacy_answer_from_evidence(
    *,
    question: str,
    user_id: str,
    documents: list[Document],
    llm: Any,
    graph_context: str,
    history_section: str,
    intent_constraints: str,
    plain_mode: bool,
    enable_visual_verification: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
    image_encoder: Callable[[str], Optional[str]] = _encode_image,
) -> GeneratedRagAnswer:
    """Generate the v8 answer projection from already retrieved evidence.

    Retrieval, corrective retrieval, and graph location are intentionally absent
    from this function.  The visual loop remains here because it is part of the
    historical answer-generation behavior and is not a v9 capability.
    """
    source_doc_ids = legacy_source_doc_ids(documents)
    doc_id_to_name: dict[str, str] = {}
    if source_doc_ids:
        try:
            doc_id_to_name = await fetch_document_filenames(source_doc_ids)
        except Exception as error:  # noqa: BLE001 - preserve best-effort labels
            logger.warning("Failed to fetch filenames from DB: %s", error)

    chunks_by_doc: dict[str, list[str]] = {}
    image_paths: set[str] = set()
    for document in documents:
        doc_id = get_document_id(document.metadata)
        if doc_id and doc_id not in doc_id_to_name:
            doc_id_to_name[doc_id] = (
                document.metadata.get("file_name")
                or document.metadata.get("source_file")
                or f"文件-{doc_id[:8]}"
            )
        source_label = doc_id_to_name.get(doc_id, "未知來源") if doc_id else "未知來源"
        chunks = chunks_by_doc.setdefault(source_label, [])
        if document.metadata.get("source", "text") == "image":
            image_path = document.metadata.get("image_path")
            if image_path and os.path.exists(image_path):
                image_paths.add(image_path)
                if document.page_content:
                    chunks.append(f"[圖片摘要] (路徑: {image_path.replace('\\\\', '/')})\n{document.page_content}")
            elif document.page_content:
                chunks.append(f"[圖片摘要] {document.page_content}")
        elif document.page_content:
            chunks.append(document.page_content)

    text_context = [
        f"=== 來源文件：{filename} ===\n（以下內容僅來自此文件，請勿與其他文件混淆）\n\n"
        + "\n\n".join(chunks)
        for filename, chunks in chunks_by_doc.items()
    ]
    image_list = list(image_paths)[:3]
    encoded_images = [
        image
        for image in await asyncio.gather(
            *(run_in_threadpool(image_encoder, image_path) for image_path in image_list)
        )
        if image
    ] if image_list else []
    context_text = "\n\n---\n\n".join(text_context) if text_context else "(無文字背景資訊)"
    graph_section = f"\n{graph_context}\n" if graph_context else ""
    if plain_mode:
        prompt_text = format_prompt(
            "plain_rag_answer",
            context_text=context_text,
            graph_section=graph_section,
            history_section=history_section,
            question=question,
            intent_constraints=intent_constraints,
        )
    else:
        prompt_text = format_prompt(
            "advanced_rag_answer",
            context_text=context_text,
            graph_section=graph_section,
            history_section=history_section,
            question=question,
            intent_constraints=intent_constraints,
            visual_instruction=(format_prompt("visual_tool_instruction") if enable_visual_verification and image_paths else ""),
        )
    message_content: list[Any] = [{"type": "text", "text": prompt_text}]
    message_content.extend(
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        for image in encoded_images
    )

    try:
        await _emit_progress(progress_callback, "answer_generation", {"image_count": len(encoded_images), "document_count": len(documents)})
        with llm_accounting_phase("answer_generation"):
            response = await llm.ainvoke([HumanMessage(content=message_content)])
        answer = response.content
        visual_meta: dict[str, Any] = {
            "visual_verification_attempted": False,
            "visual_tool_call_count": 0,
            "visual_force_fallback_used": False,
        }
        if enable_visual_verification and not image_paths:
            visual_meta["visual_not_applicable"] = True
        tool_calls: list[dict[str, Any]] = []
        if enable_visual_verification and image_paths:
            answer, tool_calls, visual_meta = await _execute_legacy_visual_verification_loop(
                initial_response=answer,
                context=context_text,
                question=question,
                user_id=user_id,
                llm=llm,
                image_paths=image_list,
                force_once_if_not_triggered=True,
            )
        return GeneratedRagAnswer(
            answer=answer,
            usage=get_llm_usage_metrics(response),
            thought_process=prompt_text,
            tool_calls=tool_calls,
            visual_verification_meta=visual_meta,
        )
    except (RuntimeError, ValueError, OSError) as error:
        logger.error("LLM error for user %s: %s", user_id, error, exc_info=True)
        return GeneratedRagAnswer(answer="抱歉，處理您的問題時發生錯誤。")


__all__ = [
    "generate_legacy_answer_from_evidence",
    "legacy_source_doc_ids",
    "parse_legacy_visual_tool_request",
]
