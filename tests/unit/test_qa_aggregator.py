"""
Unit tests for the Quality Assurance Aggregator and reporting consolidation.
"""

from datetime import datetime, timezone
import pytest

from evaluation.aggregators.qa_aggregator import EvaluationAggregator
from evaluation.models import (
    EvaluationResult,
    ExecutionResult,
    MetricDetail,
    MetricResult,
)


@pytest.fixture
def aggregator() -> EvaluationAggregator:
    """Fixture providing an EvaluationAggregator instance."""
    return EvaluationAggregator()


@pytest.fixture
def base_exec_result() -> ExecutionResult:
    """Fixture providing a dummy ExecutionResult for aggregation testing."""
    return ExecutionResult(
        conversation_id="AGG_TEST_001",
        scenario_name="Test Scenario",
        recognized_text="Hello",
        assistant_response="Hi there",
        conversation_history=[],
        audio_input_path="test_in.wav",
        audio_output_path="test_out.wav",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.mark.smoke
@pytest.mark.functional
def test_aggregator_empty_evaluators(aggregator: EvaluationAggregator, base_exec_result: ExecutionResult) -> None:
    """Test aggregation when an empty evaluator results list is provided."""
    rep = aggregator.aggregate(base_exec_result, [])
    assert rep.overall_score == 0.0
    assert rep.pass_rate == 0.0
    assert rep.passed is False
    assert "No evaluators registered" in rep.warnings[0]


@pytest.mark.functional
def test_aggregator_all_passing(aggregator: EvaluationAggregator, base_exec_result: ExecutionResult) -> None:
    """Test aggregation across multiple passing evaluators with sub-metrics."""
    m1 = MetricResult(metric_name="M1", score=0.9, threshold=0.7, passed=True)
    m2 = MetricResult(metric_name="M2", score=0.95, threshold=0.8, passed=True)

    ev1 = EvaluationResult(
        evaluator_name="Eval1",
        overall_score=0.925,
        passed=True,
        metric_results=[m1, m2],
        execution_time_ms=50.0,
    )

    ev2 = EvaluationResult(
        evaluator_name="Eval2",
        overall_score=0.95,
        passed=True,
        metric_results=[],
        execution_time_ms=30.0,
    )

    rep = aggregator.aggregate(base_exec_result, [ev1, ev2])
    assert rep.passed is True
    assert len(rep.evaluator_results) == 2
    assert rep.overall_score == (0.925 + 0.95) / 2.0
    assert rep.pass_rate == 1.0
    assert len(rep.failed_metrics) == 0


@pytest.mark.functional
def test_aggregator_with_failures_and_warnings(aggregator: EvaluationAggregator, base_exec_result: ExecutionResult) -> None:
    """Test aggregation when metrics fail and warnings/errors are reported."""
    m_pass = MetricResult(metric_name="GoodMetric", score=0.9, threshold=0.7, passed=True)
    m_fail = MetricResult(metric_name="BadMetric", score=0.4, threshold=0.8, passed=False)

    ev1 = EvaluationResult(
        evaluator_name="FailingEval",
        overall_score=0.65,
        passed=False,
        metric_results=[m_pass, m_fail],
        warnings=["Tone too informal"],
        recommendations=["Use polite language"],
        execution_time_ms=100.0,
    )

    ev2 = EvaluationResult(
        evaluator_name="ErrorEval",
        overall_score=0.0,
        passed=False,
        error="API timeout during eval",
        execution_time_ms=10.0,
    )

    rep = aggregator.aggregate(base_exec_result, [ev1, ev2])
    assert rep.passed is False
    assert rep.pass_rate == 0.5  # 1 pass out of 2 total sub-metrics
    assert len(rep.failed_metrics) == 1
    assert "BadMetric" in rep.failed_metrics[0]
    assert any("API timeout" in w for w in rep.warnings)
    assert any("Tone too informal" in w for w in rep.warnings)
    assert any("Use polite language" in r for r in rep.recommendations)
