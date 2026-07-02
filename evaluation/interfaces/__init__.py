"""
Consolidated Interfaces Package for the AI Evaluation Framework.
"""

from evaluation.interfaces.evaluator import Evaluator
from evaluation.interfaces.metrics import (
    BaseConversationValidator,
    BaseMetricEvaluator,
    BaseSecurityEvaluator,
    BaseVoiceEvaluator,
)
from evaluation.interfaces.reporting import BaseReportGenerator
from evaluation.interfaces.runner import BaseTestRunner

__all__ = [
    "Evaluator",
    "BaseMetricEvaluator",
    "BaseVoiceEvaluator",
    "BaseConversationValidator",
    "BaseSecurityEvaluator",
    "BaseTestRunner",
    "BaseReportGenerator",
]
