"""
Latency Metric Evaluator.

Measures and validates execution latencies against SLA thresholds for:
- Speech Recognition (STT) latency
- LLM response latency
- Speech Generation (TTS) latency
- Total end-to-end voice roundtrip latency
"""

import time
from typing import List
from loguru import logger

from evaluation.config import eval_settings
from evaluation.models import MetricResult


class LatencyEvaluator:
    """
    Evaluator for voice pipeline execution latencies against Service Level Agreements (SLAs).
    """

    def __init__(
        self,
        stt_threshold: float = eval_settings.EVAL_MAX_STT_LATENCY_SEC,
        llm_threshold: float = eval_settings.EVAL_MAX_LLM_LATENCY_SEC,
        tts_threshold: float = eval_settings.EVAL_MAX_TTS_LATENCY_SEC,
        total_threshold: float = eval_settings.EVAL_MAX_TOTAL_LATENCY_SEC,
    ) -> None:
        """Initialize latency SLAs."""
        self.stt_threshold = stt_threshold
        self.llm_threshold = llm_threshold
        self.tts_threshold = tts_threshold
        self.total_threshold = total_threshold

    def evaluate_latencies(
        self,
        stt_latency: float,
        llm_latency: float,
        tts_latency: float,
        total_latency: float,
    ) -> List[MetricResult]:
        """
        Evaluate all pipeline latencies against configured SLA limits.

        Args:
            stt_latency: Speech-to-Text latency in seconds.
            llm_latency: LLM generation latency in seconds.
            tts_latency: Text-to-Speech latency in seconds.
            total_latency: Total turn execution time in seconds.

        Returns:
            List of MetricResult records for each stage and total roundtrip latency.
        """
        start_time = time.perf_counter()
        results: List[MetricResult] = []

        # 1. STT Latency
        stt_passed = stt_latency <= self.stt_threshold
        stt_score = 1.0 if stt_passed else max(0.0, 1.0 - (stt_latency - self.stt_threshold) / self.stt_threshold)
        stt_reason = f"STT Latency: {stt_latency:.3f}s (SLA: <= {self.stt_threshold:.3f}s)."
        if not stt_passed:
            stt_reason += f" Exceeded STT SLA limit by {(stt_latency - self.stt_threshold):.3f}s."
        results.append(
            MetricResult(
                metric_name="STT Latency SLA",
                score=stt_score,
                threshold=self.stt_threshold,
                passed=stt_passed,
                reason=stt_reason,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                details={"latency_sec": stt_latency, "stage": "STT"},
            )
        )

        # 2. LLM Latency
        llm_passed = llm_latency <= self.llm_threshold
        llm_score = 1.0 if llm_passed else max(0.0, 1.0 - (llm_latency - self.llm_threshold) / self.llm_threshold)
        llm_reason = f"LLM Latency: {llm_latency:.3f}s (SLA: <= {self.llm_threshold:.3f}s)."
        if not llm_passed:
            llm_reason += f" Exceeded LLM SLA limit by {(llm_latency - self.llm_threshold):.3f}s."
        results.append(
            MetricResult(
                metric_name="LLM Latency SLA",
                score=llm_score,
                threshold=self.llm_threshold,
                passed=llm_passed,
                reason=llm_reason,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                details={"latency_sec": llm_latency, "stage": "LLM"},
            )
        )

        # 3. TTS Latency
        tts_passed = tts_latency <= self.tts_threshold
        tts_score = 1.0 if tts_passed else max(0.0, 1.0 - (tts_latency - self.tts_threshold) / self.tts_threshold)
        tts_reason = f"TTS Latency: {tts_latency:.3f}s (SLA: <= {self.tts_threshold:.3f}s)."
        if not tts_passed:
            tts_reason += f" Exceeded TTS SLA limit by {(tts_latency - self.tts_threshold):.3f}s."
        results.append(
            MetricResult(
                metric_name="TTS Latency SLA",
                score=tts_score,
                threshold=self.tts_threshold,
                passed=tts_passed,
                reason=tts_reason,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                details={"latency_sec": tts_latency, "stage": "TTS"},
            )
        )

        # 4. Total Latency
        total_passed = total_latency <= self.total_threshold
        total_score = 1.0 if total_passed else max(0.0, 1.0 - (total_latency - self.total_threshold) / self.total_threshold)
        total_reason = f"Total End-to-End Latency: {total_latency:.3f}s (SLA: <= {self.total_threshold:.3f}s)."
        if not total_passed:
            total_reason += f" Exceeded Total SLA limit by {(total_latency - self.total_threshold):.3f}s."
        results.append(
            MetricResult(
                metric_name="Total Latency SLA",
                score=total_score,
                threshold=self.total_threshold,
                passed=total_passed,
                reason=total_reason,
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                details={"latency_sec": total_latency, "stage": "Total"},
            )
        )

        logger.info(
            f"Latencies Evaluated | STT: {stt_latency:.2f}s | LLM: {llm_latency:.2f}s | TTS: {tts_latency:.2f}s | Total: {total_latency:.2f}s"
        )
        return results
