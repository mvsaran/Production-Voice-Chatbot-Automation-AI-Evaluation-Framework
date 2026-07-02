"""
Unified AI Evaluation Domain Models Package.

Exports all data models, schemas, and backward-compatibility type aliases for
scenarios, metric results, execution artifacts, and enterprise QA reports.
"""

from evaluation.models.scenarios import (
    ConversationScenario,
    ScenarioTurn,
)
from evaluation.models.results import (
    ExecutionResult,
    MetricDetail,
    MetricResult,
    MetricStatusEnum,
    SecurityEvaluationResult,
    TurnExecutionArtifacts,
    VoiceQualityResult,
    ConversationValidationResult,
)
from evaluation.models.reports import (
    AggregatedEvaluationReport,
    ConversationEvaluationReport,
    DashboardReadyEvaluationOutput,
    EvaluationResult,
)

__all__ = [
    "ConversationScenario",
    "ScenarioTurn",
    "ExecutionResult",
    "MetricDetail",
    "MetricResult",
    "MetricStatusEnum",
    "SecurityEvaluationResult",
    "TurnExecutionArtifacts",
    "VoiceQualityResult",
    "ConversationValidationResult",
    "AggregatedEvaluationReport",
    "ConversationEvaluationReport",
    "DashboardReadyEvaluationOutput",
    "EvaluationResult",
]
