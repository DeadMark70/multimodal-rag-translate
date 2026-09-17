# Evaluation inference and pricing

## Controls and persistence

Campaigns freeze `ragas_service_tier` (standard/flex),
`ragas_request_timeout_seconds` (default 900), `ragas_max_attempts` (default 5,
including the first call), `ragas_standard_fallback` (default false), and
`ragas_rpm_limit`. These affect the evaluator only. `EVALUATION_EVALUATOR_MODEL`
selects the evaluator for new work; each item retains that model in its snapshot.
Historical snapshots without these fields use Standard.

The production worker saves each answer/metric independently. The
`ragas_parallel_batches` setting (1–8) controls active metrics and provider
requests, including the three relevance samples. `ragas_batch_size` is retained
as a legacy API field and removed from UI controls. Up to 64 items are prefetched
with heartbeats maintained for queued/active claims.

Only the provider layer retries transient calls. SDK and RAGAS transport attempts
are each one; retries count toward RPM and accounting. The UI reports provider
retrying without releasing the active durable claim. Exhausted items retain all
completed scores and can be rerun through the partial-rerun UI. Cancellation and
recovery keep the existing durable ledger.

Flex has variable latency. API requests have configurable timeouts; the enclosing
metric timeout includes multiple generations/repairs. LangChain accepts seconds
and converts them to SDK milliseconds. Standard fallback requires opt-in and
runs only after Flex capacity retries are exhausted; its tier is recorded.

## Dependencies

Use google-genai 1.75.0, langchain-google-genai 4.2.0 and ragas 0.4.3. SDK 1.75
already supports Flex. SDK 2.x conflicts with marker-pdf 1.10.1. Marker 2.x also
changes OCR/Transformers/OpenAI dependencies, so that migration is deferred.
Request-level tests verify Flex actually reaches GenerateContentConfig.

## Cache and usage

Cache metadata uses existing raw usage JSON. Analytics reports cached/input
token ratio, request hit ratio and measurement coverage for answering and
scoring separately. Missing data is N/A; cached input is a subset of input.
LangChain output includes thinking; normalization removes that overlap.

RAGAS already puts fixed instructions, schemas and examples before variable
input. Prompts remain unchanged; compatible metric work is grouped together.
There is no claim of a measured hit-rate increase, prompt padding, answer reuse,
explicit cache, or embedding cache.

## Prices and deployment

Authenticated GET `/api/evaluation/pricing` returns freshness and evaluator
model. POST `/api/evaluation/pricing/refresh` fetches official prices. The app
also refreshes daily; fake-provider lifespans skip that network work.

The parser handles supported Gemini 2.5/3.x text model prices, Standard/Flex,
cached input, thinking-inclusive output, context thresholds and effective dates.
Unknown units are not guessed. Failed refreshes keep the previous snapshot;
unknown model/tier prices are N/A and never block evaluation.

Snapshots live in `output/evaluation-prices`, overridable with
`EVALUATION_PRICE_DIRECTORY`. Preserve this on the existing output volume.
Snapshot files are immutable and current.json is replaced atomically. Usage
events retain their price snapshot ID and original estimate. No DB migration or
new service is needed. Rebuild both frontend and backend.

`EVALUATION_PRICE_SNAPSHOT_PATH` remains a manual override and disables automatic
refresh. Manual files need explicit cache/tier rates for discounted calls.
Prices are public text-token estimates, not account invoices or tool, grounding,
or explicit-cache storage charges. No spend cap is introduced.

Sources: [Flex](https://ai.google.dev/gemini-api/docs/generate-content/flex-inference),
[caching](https://ai.google.dev/gemini-api/docs/generate-content/caching),
[prices](https://ai.google.dev/gemini-api/docs/pricing).
