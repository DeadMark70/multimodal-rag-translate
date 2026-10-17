"""Adapt the shared V10 result to ordinary chat without the legacy planner."""

from core.llm_factory import llm_runtime_override
from data_base.agentic_chat_service import build_query_resolution_question
from data_base.agentic_v10.subquery_pipeline_service import AgenticV10PipelineService
from data_base.rag_pipeline_schemas import ProgressCallback, RAGResult
from data_base.schemas import AskRequest


async def answer_agentic_chat(
    request: AskRequest,
    user_id: str,
    *,
    progress_callback: ProgressCallback | None = None,
) -> RAGResult:
    # Match the evaluation preset default instead of the legacy planner's 512
    # and synthesizer's 2048-token limits, which also count thinking tokens.
    with llm_runtime_override(max_output_tokens=8192):
        result = await AgenticV10PipelineService().execute(
            question=request.question,
            user_id=user_id,
            authorized_doc_ids=request.doc_ids,
            decomposition_question=build_query_resolution_question(
                question=request.question,
                history=[
                    {"role": message.role.value, "content": message.content}
                    for message in request.history or []
                ],
            ),
            progress_callback=progress_callback,
        )
    if not result.answer.strip():
        raise RuntimeError("V10 did not produce an answer")

    # Reference numbers can have gaps after V10 filters its initial evidence.
    # Keep their original IDs instead of renumbering the retained source list.
    references = result.agent_trace["agentic_v10"]["context_pack"][
        "final_reference_ids"
    ]
    documents = [
        document.model_copy(
            update={"metadata": {**document.metadata, "reference_id": reference}}
        )
        for document, reference in zip(result.documents, references, strict=True)
    ]
    return result._replace(documents=documents)
