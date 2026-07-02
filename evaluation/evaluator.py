"""
Backward-compatibility re-export shim for EvaluationAggregator.
Please import from `evaluation.aggregators.qa_aggregator` in new code.
"""

from evaluation.aggregators.qa_aggregator import EvaluationAggregator

__all__ = ["EvaluationAggregator"]
