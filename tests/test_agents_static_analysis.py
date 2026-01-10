import inspect
import pytest
from agents import planner, evaluator, synthesizer

def test_planner_structure():
    """Verify Planner module structure."""
    assert hasattr(planner, "TaskPlanner")
    assert hasattr(planner, "ResearchPlan")
    assert hasattr(planner, "SubTask")
    
    # Check method signatures
    sig = inspect.signature(planner.TaskPlanner.plan)
    assert "question" in sig.parameters
    
    sig = inspect.signature(planner.TaskPlanner.create_followup_tasks)
    assert "current_findings" in sig.parameters

def test_evaluator_structure():
    """Verify Evaluator module structure."""
    assert hasattr(evaluator, "RAGEvaluator")
    assert hasattr(evaluator, "EvaluationResult")
    assert hasattr(evaluator, "DetailedEvaluationResult")
    
    # Check method signatures
    sig = inspect.signature(evaluator.RAGEvaluator.evaluate_detailed)
    assert "documents" in sig.parameters
    assert "answer" in sig.parameters

def test_synthesizer_structure():
    """Verify Synthesizer module structure."""
    assert hasattr(synthesizer, "ResultSynthesizer")
    assert hasattr(synthesizer, "ResearchReport")
    
    # Check method signatures
    sig = inspect.signature(synthesizer.ResultSynthesizer.synthesize)
    assert "sub_results" in sig.parameters

def test_error_handling_existence():
    """
    Static check for try-except blocks in key methods.
    This is a heuristic check using source code inspection.
    """
    modules = [planner, evaluator, synthesizer]
    
    for module in modules:
        source = inspect.getsource(module)
        # Check if try/except is used (basic check)
        assert "try:" in source, f"Module {module.__name__} missing try/except blocks"
        assert "except" in source, f"Module {module.__name__} missing except blocks"

def test_type_hints():
    """
    Check if type hints are used in __init__ methods.
    """
    classes = [
        planner.TaskPlanner,
        evaluator.RAGEvaluator,
        synthesizer.ResultSynthesizer
    ]
    
    for cls in classes:
        init_sig = inspect.signature(cls.__init__)
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            assert param.annotation != inspect.Parameter.empty, f"{cls.__name__}.__init__ argument '{param_name}' missing type hint"
