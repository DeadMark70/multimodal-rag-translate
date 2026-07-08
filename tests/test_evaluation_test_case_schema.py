from evaluation.dataset_generator import build_ragas_ready_dataset
from evaluation.router import TestCaseCreateRequest as CreateRequestSchema
from evaluation.router import TestCaseImportRequest as ImportRequestSchema
from evaluation.schemas import TestCase as CaseSchema


def _research_case_payload() -> dict:
    return {
        "id": "Q09",
        "question": "What evidence supports the result?",
        "ground_truth": "Answer",
        "question_version": "v2.0.0",
        "difficulty": "very_hard",
        "required_modalities": ["text", "table"],
        "atomic_facts": [
            {
                "atomic_fact_id": "Q09-F1",
                "fact_text": "The reported value is 0.9079.",
                "required_doc_id": "paper-a.pdf",
                "required_page": 5,
            }
        ],
        "expected_evidence": [
            {
                "evidence_id": "Q09-E1",
                "doc_id": "paper-a.pdf",
                "page": 5,
                "modality": "table",
            }
        ],
    }


def test_test_case_accepts_research_metadata_and_normalizes_difficulty() -> None:
    case = CaseSchema(**_research_case_payload())
    spaced_case = CaseSchema(**{**_research_case_payload(), "difficulty": "Very Hard"})

    assert case.question_version == "v2.0.0"
    assert case.difficulty == "very-hard"
    assert spaced_case.difficulty == "very-hard"
    assert case.required_modalities == ["text", "table"]
    assert case.atomic_facts[0]["atomic_fact_id"] == "Q09-F1"
    assert case.expected_evidence[0]["evidence_id"] == "Q09-E1"


def test_test_case_request_schemas_preserve_research_metadata() -> None:
    create_request = CreateRequestSchema.model_validate(_research_case_payload())

    dumped_create = create_request.model_dump(exclude_none=True)
    assert dumped_create["question_version"] == "v2.0.0"
    assert dumped_create["difficulty"] == "very-hard"
    assert dumped_create["required_modalities"] == ["text", "table"]
    assert dumped_create["atomic_facts"][0]["atomic_fact_id"] == "Q09-F1"
    assert dumped_create["expected_evidence"][0]["evidence_id"] == "Q09-E1"

    import_request = ImportRequestSchema.model_validate(
        {"metadata": {}, "questions": [_research_case_payload()]}
    )
    dumped_import = import_request.questions[0].model_dump(exclude_none=True)
    assert dumped_import["question_version"] == "v2.0.0"
    assert dumped_import["required_modalities"] == ["text", "table"]


def test_ragas_ready_dataset_preserves_research_metadata() -> None:
    ready = build_ragas_ready_dataset(
        {
            "metadata": {"dataset_version": "2.0.0"},
            "questions": [_research_case_payload()],
        },
        derived_from="master.json",
    )

    question = ready["questions"][0]
    assert question["question_version"] == "v2.0.0"
    assert question["difficulty"] == "very-hard"
    assert question["required_modalities"] == ["text", "table"]
    assert question["atomic_facts"][0]["atomic_fact_id"] == "Q09-F1"
    assert question["expected_evidence"][0]["evidence_id"] == "Q09-E1"
