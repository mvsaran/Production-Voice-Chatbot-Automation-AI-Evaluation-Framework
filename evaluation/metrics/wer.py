"""
Word Error Rate (WER) Metric Evaluator.

Computes Speech-To-Text accuracy by calculating the Word Error Rate (Levenshtein distance
at the word level) between expected ground-truth text and Whisper recognized text using `jiwer`.
"""

import time
from typing import Any, List, Optional
import jiwer
from loguru import logger

from evaluation.config import eval_settings
from evaluation.interfaces.metrics import BaseMetricEvaluator
from evaluation.models import MetricResult


class WEREvaluator(BaseMetricEvaluator):
    """
    Evaluator for Word Error Rate (WER) using jiwer.
    Measures transcription accuracy against configured SLA thresholds.
    """

    def __init__(self, threshold: Optional[float] = None) -> None:
        """Initialize WEREvaluator with optional custom threshold override."""
        self._threshold = threshold if threshold is not None else eval_settings.EVAL_MAX_WER

    async def evaluate(
        self,
        prompt: str,
        response: str,
        expected_response: Optional[str] = None,
        context: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Evaluate WER between expected text and recognized text.

        Args:
            prompt: Fallback reference text if expected_response is not provided.
            response: Recognized text from speech-to-text service.
            expected_response: Ground-truth reference text.
            context: Optional dialogue context (unused for WER).
            kwargs: Additional arguments.

        Returns:
            MetricResult with WER score, pass/fail status, and diagnostic reasoning.
        """
        start_time = time.perf_counter()
        reference = (expected_response if expected_response is not None else prompt).strip()
        hypothesis = response.strip()

        logger.debug(f"Computing WER | Reference: '{reference}' | Hypothesis: '{hypothesis}'")

        if not reference and not hypothesis:
            wer_score = 0.0
            reason = "Both reference and hypothesis are empty strings; WER is 0.0 (perfect match)."
        elif not reference or not hypothesis:
            wer_score = 1.0
            reason = f"One of reference or hypothesis is empty (Ref len: {len(reference)}, Hyp len: {len(hypothesis)}); WER is 1.0."
        else:
            try:
                wer_val = jiwer.wer(reference, hypothesis)
                wer_score = float(wer_val)
                reason = f"WER computed as {wer_score:.4f} (Threshold: <= {self._threshold:.4f})."
            except Exception as e:
                logger.error(f"Error computing WER with jiwer: {str(e)}")
                return MetricResult(
                    metric_name="Word Error Rate (WER)",
                    score=0.0,
                    threshold=self._threshold,
                    passed=False,
                    reason=f"Failed to compute WER: {str(e)}",
                    error_message=str(e),
                    execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                )

        passed = wer_score <= self._threshold
        # Note: For accuracy score representation in UI/QA, 1.0 - min(1.0, wer_score) is stored as the normalized accuracy score,
        # but here we store accuracy score (1.0 - wer) or wer directly. To maintain standard pass/fail check:
        # We store score as recognition accuracy (max(0.0, 1.0 - wer_score)) so higher score = better quality.
        accuracy_score = max(0.0, 1.0 - wer_score)

        if not passed:
            reason += f" Breached maximum allowed WER threshold of {self._threshold:.4f}."

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(f"WER Evaluated | Score: {accuracy_score:.4f} (WER: {wer_score:.4f}) | Passed: {passed}")

        return MetricResult(
            metric_name="Word Error Rate (WER)",
            score=accuracy_score,
            threshold=self._threshold,
            passed=passed,
            reason=reason,
            execution_time_ms=execution_time_ms,
            details={"wer": wer_score, "reference": reference, "hypothesis": hypothesis},
        )
