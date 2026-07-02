"""
Abstract Evaluator Interface.

Defines the standard protocol contract that all evaluation engines (DeepEval,
RAGAS, OpenAI Evals, Custom Enterprise Metrics) must implement.
"""

from abc import ABC, abstractmethod
from evaluation.models import EvaluationResult, ExecutionResult


class Evaluator(ABC):
    """Abstract interface for AI Quality Assurance and Safety Evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier name of the evaluation engine."""
        pass

    @abstractmethod
    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        """
        Evaluate an end-to-end voice chatbot execution result.

        Args:
            execution_result: Immutable ExecutionResult containing transcribed text,
                              LLM response, latencies, and conversation history.

        Returns:
            EvaluationResult detailing metric scores, thresholds, pass/fail status,
            warnings, and recommendations.
        """
        pass
