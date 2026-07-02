"""
Backward-compatibility re-export shim for execution results and evaluation reports.
Please import directly from `evaluation.models` in new code.
"""

from evaluation.models import (
    AggregatedEvaluationReport,
    EvaluationResult,
    ExecutionResult,
    MetricDetail,
    MetricResult,
)

__all__ = [
    "AggregatedEvaluationReport",
    "EvaluationResult",
    "ExecutionResult",
    "MetricDetail",
    "MetricResult",
]
