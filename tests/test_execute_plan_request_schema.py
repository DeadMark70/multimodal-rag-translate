import pytest
from pydantic import ValidationError
from data_base.schemas_deep_research import ExecutePlanRequest, EditableSubTask

def test_execute_plan_request_accepts_conversation_id():
    """
    Test that ExecutePlanRequest accepts conversation_id.
    This test is expected to fail before the schema is updated.
    """
    sub_tasks = [
        EditableSubTask(id=1, question="Test?", enabled=True)
    ]
    
    # This should NOT raise ValidationError if conversation_id is allowed
    try:
        request = ExecutePlanRequest(
            original_question="Test?",
            sub_tasks=sub_tasks,
            conversation_id="some-uuid-string"
        )
        assert request.conversation_id == "some-uuid-string"
    except ValidationError as e:
        pytest.fail(f"ExecutePlanRequest did not accept conversation_id: {e}")
    except AttributeError:
        pytest.fail("ExecutePlanRequest has no attribute conversation_id")

if __name__ == "__main__":
    test_execute_plan_request_accepts_conversation_id()
