"""
Character Error Rate (CER) Metric Evaluator.

Computes character-level transcription accuracy by calculating the Character Error Rate
between expected text and Whisper recognized text using `jiwer`.
"""

import time
from typing import Any, List, Optional
import jiwer
from loguru import logger

from evaluation.config import eval_settings
from evaluation.interfaces.metrics import BaseMetricEvaluator
from evaluation.models import MetricResult


class CEREvaluator(BaseMetricEvaluator):
    """
    Evaluator for Character Error Rate (CER) using jiwer.
    Measures character-level transcription accuracy against configured SLA thresholds.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        """Initialize CEREvaluator with optional custom threshold override."""
        self._threshold = threshold if threshold is not None else eval_settings.EVAL_MAX_CER

    async def evaluate(
        self,
        prompt: str,
        response: str,
        expected_response: Optional[str] = None,
        context: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Evaluate CER between expected text and recognized text.

        Args:
            prompt: Fallback reference text if expected_response is not provided.
            response: Recognized text from speech-to-text service.
            expected_response: Ground-truth reference text.
            context: Optional dialogue context (unused for CER).
            kwargs: Additional arguments.

        Returns:
            MetricResult with CER score, pass/fail status, and diagnostic reasoning.
        """
        start_time = time.perf_counter()
        reference = (expected_response if expected_response is not None else prompt).strip()
        hypothesis = response.strip()

        logger.debug(f"Computing CER | Reference: '{reference}' | Hypothesis: '{hypothesis}'")

        if not reference and not hypothesis:
            cer_score = 0.0
            reason = "Both reference and hypothesis are empty strings; CER is 0.0 (perfect match)."
        elif not reference or not hypothesis:
            cer_score = 1.0
            reason = f"One of reference or hypothesis is empty (Ref len: {len(reference)}, Hyp len: {len(hypothesis)}); CER is 1.0."
        else:
            try:
                cer_val = jiwer.cer(reference, hypothesis)
                cer_score = float(cer_val)
                reason = f"CER computed as {cer_score:.4f} (Threshold: <= {self._threshold:.4f})."
            except Exception as e:
                logger.error(f"Error computing CER with jiwer: {str(e)}")
                return MetricResult(
                    metric_name="Character Error Rate (CER)",
                    score=0.0,
                    threshold=self._threshold,
                    passed=False,
                    reason=f"Failed to compute CER: {str(e)}",
                    error_message=str(e),
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                )

        passed = cer_score <= self._threshold
        accuracy_score = max(0.0, 1.0 - cer_score)

        if not passed:
            reason += f" Breached maximum allowed CER threshold of {self._threshold:.4f}."

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"CER Evaluated | Score: {accuracy_score:.4f} (CER: {cer_score:.4f}) | Passed: {passed}")

        return MetricResult(
            metric_name="Character Error Rate (CER)",
            score=accuracy_score,
            threshold=self._threshold,
            passed=passed,
            reason=reason,
            execution_time_ms=execution_time_ms,
            details={"cer": cer_score, "reference": reference, "hypothesis": hypothesis},
        )
