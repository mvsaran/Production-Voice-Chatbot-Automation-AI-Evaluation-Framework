"""
Evaluation QA Report Data Models.

Defines immutable Pydantic v2 schemas for evaluator plugin results (`EvaluationResult`),
consolidated multi-evaluator QA reports (`AggregatedEvaluationReport`), and
zero-code UI dashboard telemetry (`DashboardReadyEvaluationOutput`).
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from evaluation.models.results import ExecutionResult, MetricResult, MetricStatusEnum


class EvaluationResult(BaseModel):
    """Immutable result payload returned by a registered Evaluator (e.g., DeepEval, RAGAS)."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    evaluator_name: str = Field(..., description="Unique identifier name of the evaluation engine.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Aggregate quality score for this evaluator.")
    passed: bool = Field(..., description="True if all metrics within this evaluator satisfied SLA thresholds.")
    metric_results: List[MetricResult] = Field(default_factory=list, description="Detailed breakdowns per evaluated metric.")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered during evaluation.")
    recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations to fix failed metrics.")
    execution_time_ms: float = Field(default=0.0, description="Time taken by the evaluator in milliseconds.")
    error: Optional[str] = Field(default=None, description="Fatal error message if evaluator failed unexpectedly.")


class AggregatedEvaluationReport(BaseModel):
    """Consolidated QA report aggregating results from all registered evaluators."""
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conversation_id: str = Field(..., description="Target scenario session UUID.")
    scenario_name: str = Field(..., description="Title of the executed test scenario.")
    execution_result: ExecutionResult = Field(..., description="Artifacts captured during pipeline execution.")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Grand average QA quality score across all evaluators.")
    pass_rate: float = Field(..., ge=0.0, le=1.0, description="Percentage of evaluated metrics that passed.")
    passed: bool = Field(..., description="True if overall pass rate is 100% and no fatal errors occurred.")
    evaluator_results: List[EvaluationResult] = Field(default_factory=list, description="Individual evaluator plugins output.")
    failed_metrics: List[str] = Field(default_factory=list, description="Summary list of metrics that breached SLA limits.")
    warnings: List[str] = Field(default_factory=list, description="Combined warning log across stages and evaluators.")
    recommendations: List[str] = Field(default_factory=list, description="Consolidated actionable improvement recommendations.")
    total_evaluation_time_ms: float = Field(default=0.0, description="Total execution and evaluation time in milliseconds.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when evaluation report was compiled."
    )


class DashboardReadyEvaluationOutput(BaseModel):
    """
    Structured JSON payload formatted specifically for UI dashboard ingestion.
    Conforms strictly to project Requirement 10 without requiring visualizer code changes.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conversation_id: str = Field(..., description="Conversation or scenario session ID.")
    prompt: str = Field(..., description="User input prompt or transcribed text.")
    response: str = Field(..., description="Chatbot generated response.")
    correctness: float = Field(default=0.0, description="Answer Correctness metric score.")
    hallucination: float = Field(default=0.0, description="Hallucination metric score.")
    faithfulness: float = Field(default=0.0, description="Faithfulness metric score.")
    wer: float = Field(default=0.0, description="Word Error Rate.")
    cer: float = Field(default=0.0, description="Character Error Rate.")
    latency: float = Field(default=0.0, description="Total end-to-end execution latency in seconds.")
    status: MetricStatusEnum = Field(..., description="Overall test execution status (PASS, FAIL, ERROR).")
    failure_reason: Optional[str] = Field(default=None, description="Explanation of failure if status is FAIL or ERROR.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Execution completion timestamp."
    )
    detailed_metrics: Optional[Dict[str, MetricResult]] = Field(default=None, description="Full metric breakdowns.")


# Backward-compatibility alias
ConversationEvaluationReport = DashboardReadyEvaluationOutput
