# Agentic v9 Content-Block Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept LangChain `AIMessage` text content blocks at the Agentic v9 structured-output boundary without weakening any JSON or evidence validation.

**Architecture:** Add one response-text normalizer beside the shared Agentic v9 provider builders. Contract planning, evidence qualification, and the contract canary consume that helper before JSON decoding while preserving already-decoded mapping payloads and raw provider responses.

**Tech Stack:** Python 3.13, `langchain-core` 1.x `AIMessage`, Pydantic 2.12, pytest, Ruff.

## Global Constraints

- Preserve raw `AIMessage` responses, accounting callbacks, budget admission, and provider invocation behavior.
- Use `AIMessage.text` when available; never stringify the whole content-block list or include `extras.signature`.
- Keep the current strict JSON envelope, evidence ID, slot ID, eligibility, and source-span validation.
- Do not treat the separate contract-planner provider-invocation failure as resolved by this change.
- Do not change OpenAPI, persisted schemas, prompts, model selection, or token limits.

---

### Task 1: Normalize structured provider text once and reconnect both consumers

**Files:**
- Modify: `data_base/agentic_v9/provider_boundary.py`
- Modify: `data_base/agentic_v9/contract_planner.py:601-616`
- Modify: `data_base/agentic_v9/evidence_extractor.py:457-474`
- Modify: `scripts/agentic_v9_contract_planner_canary.py:49-72,173-199,285`
- Test: `tests/test_agentic_v9_provider_boundary.py`
- Test: `tests/test_agentic_v9_contract_planner.py`
- Test: `tests/test_agentic_v9_evidence_extractor.py`
- Test: `tests/test_agentic_v9_contract_planner_canary.py`

**Interfaces:**
- Produces: `provider_response_text(response: Any) -> str | None` in `data_base.agentic_v9.provider_boundary`.
- Consumes: a raw string, a mapping wrapper with `content`, or an object exposing LangChain's string-compatible `.text` property.
- Returns: text only; returns `None` for responses without a supported textual representation.

- [x] **Step 1: Write focused failing tests for the observed content-block response**

Add a shared-boundary test using the actual LangChain message type:

```python
from langchain_core.messages import AIMessage

from data_base.agentic_v9.provider_boundary import provider_response_text


def test_provider_response_text_uses_text_blocks_without_signature_metadata() -> None:
    response = AIMessage(
        content=[
            {
                "type": "text",
                "text": '{"packets":[]}',
                "extras": {"signature": "must-not-enter-json"},
            }
        ]
    )

    assert provider_response_text(response) == '{"packets":[]}'
    assert "must-not-enter-json" not in provider_response_text(response)
```

Add an evidence-extractor regression matching the real canary payload:

```python
@pytest.mark.asyncio
async def test_content_block_provider_response_qualifies_source_bound_packet() -> None:
    statement = "The method uses a two-stage decoder for small lesions."
    response = AIMessage(
        content=[
            {
                "type": "text",
                "text": json.dumps({
                    "packets": [{
                        "source_evidence_id": "E1",
                        "slot_ids": ["S1"],
                        "statement": "a two-stage decoder",
                    }]
                }),
                "extras": {"signature": "provider-signature"},
            }
        ]
    )
    outcome = await EvidenceExtractor(_RecordingInvoker(response)).extract_with_outcome(
        _contract(_slot("S1", "Describe the decoder architecture.")),
        [_item("E1", statement, slot_ids=["S1"])],
        repairs_complete=True,
        question="What decoder architecture is used?",
    )

    assert outcome.status == "provider_qualified"
    assert [packet.evidence_id for packet in outcome.packets] == ["curated:E1:S1"]
    assert outcome.packets[0].validation_status == "quote_bound"
```

Add production planner and canary variants whose `AIMessage.content` is the same
typed text-block list. Assert the planner produces a complete contract and the
minimal canary returns exit code `0`. Keep existing malformed-response tests.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_contract_planner_canary.py -k "content_block or response_text" -q
```

Expected: failures show the helper is missing and current planner/extractor/canary
reject list-valued `AIMessage.content`. Collection and fixtures must otherwise be valid.

- [x] **Step 3: Implement the strict shared text normalizer**

In `data_base/agentic_v9/provider_boundary.py` add:

```python
def provider_response_text(response: Any) -> str | None:
    """Return provider text without serializing non-text content blocks."""
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        content = response.get("content")
        return content if isinstance(content, str) else None
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    text = getattr(response, "text", None)
    return str(text) if isinstance(text, str) else None
```

This intentionally relies on LangChain's `.text` normalization for `AIMessage`
objects and does not manually accept arbitrary lists.

- [x] **Step 4: Route strict JSON consumers through the helper**

In `contract_planner._parse_decision`, replace direct `.content` extraction with:

```python
content = provider_response_text(response)
if not isinstance(content, str) or not content.strip():
    raise PlannerProviderEmptyResponseError
```

In `_parse_curated_packets`, retain direct `{"packets": ...}` mappings. For all
other response wrappers, call `provider_response_text`, JSON-decode the returned
string, then apply the existing exact key and semantic checks unchanged.

In the contract canary, lazily expose `provider_response_text` through
`_ProviderStack` and use it in `_response_content`. This preserves the existing
rule that invalid model config fails before importing the provider stack.

- [x] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_contract_planner_canary.py -q
```

Expected: all tests pass. The existing malformed JSON, invalid packet, provider
failure, and sanitized canary cases must remain green.

- [x] **Step 6: Run the impacted regression and lint gates**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_evidence_qualification_canary.py tests/test_agentic_v9_contract_planner_canary.py tests/test_agentic_v9_campaign_runtime.py -q
```

Run:

```powershell
.\.venv\Scripts\python.exe -m ruff check data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/evidence_extractor.py scripts/agentic_v9_contract_planner_canary.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_contract_planner_canary.py
```

Run `git diff --check` and review the complete scoped diff. OpenAPI regeneration is
not required because no HTTP or persisted schema changes.

- [x] **Step 7: Commit the independently deployable fix**

```powershell
git add -- data_base/agentic_v9/provider_boundary.py data_base/agentic_v9/contract_planner.py data_base/agentic_v9/evidence_extractor.py scripts/agentic_v9_contract_planner_canary.py tests/test_agentic_v9_provider_boundary.py tests/test_agentic_v9_contract_planner.py tests/test_agentic_v9_evidence_extractor.py tests/test_agentic_v9_contract_planner_canary.py docs/superpowers/plans/2026-08-16-agentic-v9-content-block-normalization.md
git commit -m "fix(agentic-v9): normalize provider content blocks"
```

- [ ] **Step 8: Verify the real-server boundaries after deployment**

Recreate the temporary model config after any container recreation, then run:

```powershell
docker compose exec -T -w /app backend python scripts/agentic_v9_evidence_qualification_canary.py --model-config-json /tmp/agentic-v9-model-config-3.1.json --invoke
docker compose exec -T -w /app backend python scripts/agentic_v9_contract_planner_canary.py --schema minimal --model-config-json /tmp/agentic-v9-model-config-3.1.json
docker compose exec -T -w /app backend python scripts/agentic_v9_contract_planner_canary.py --schema current --model-config-json /tmp/agentic-v9-model-config-3.1.json
```

Expected for qualification: `success=true`, `qualified_packet_count=1`, and
`semantic_qualification="provider_qualified"`. Interpret contract minimal/current
independently because their provider invocation failure predates this parsing fix.
