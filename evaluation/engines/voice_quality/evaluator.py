"""
Voice Quality Evaluation Engine.

Orchestrates Speech-to-Text accuracy metrics (WER, CER) and pipeline latency SLAs
to provide comprehensive voice quality assurance for chatbot automation.
"""

import time
from typing import List, Optional
from loguru import logger

from evaluation.config import eval_settings
from evaluation.interfaces.evaluator import Evaluator
from evaluation.interfaces.metrics import BaseVoiceEvaluator
from evaluation.metrics.cer import CEREvaluator
from evaluation.metrics.latency import LatencyEvaluator
from evaluation.metrics.wer import WEREvaluator
from evaluation.models import (
    EvaluationResult,
    ExecutionResult,
    MetricResult,
    MetricStatusEnum,
    VoiceQualityResult,
)


class VoiceQualityEvaluator(BaseVoiceEvaluator, Evaluator):
    """
    Production evaluation engine for voice quality assurance.
    Measures speech recognition accuracy (WER, CER) and enforces latency SLAs.
    """

    def __init__(
        self,
        wer_evaluator: Optional[WEREvaluator] = None,
        cer_evaluator: Optional[CEREvaluator] = None,
        latency_evaluator: Optional[LatencyEvaluator] = None,
    ) -> None:
        """Initialize VoiceQualityEvaluator with optional metric evaluators."""
        self._wer_evaluator = wer_evaluator or WEREvaluator()
        self._cer_evaluator = cer_evaluator or CEREvaluator()
        self._latency_evaluator = latency_evaluator or LatencyEvaluator()

    @property
    def name(self) -> str:
        """Return unique evaluator name."""
        return "VoiceQualityEvaluator"

    async def evaluate_voice(
        self,
        expected_text: str,
        recognized_text: str,
        stt_latency_sec: float,
        tts_latency_sec: float,
        total_latency_sec: float,
    ) -> VoiceQualityResult:
        """
        Evaluate speech recognition accuracy and SLAs for a single voice interaction.

        Args:
            expected_text: Ground truth reference text.
            recognized_text: Transcribed hypothesis text from Whisper.
            stt_latency_sec: Speech recognition latency in seconds.
            tts_latency_sec: Audio synthesis latency in seconds.
            total_latency_sec: Total roundtrip latency in seconds.

        Returns:
            VoiceQualityResult detailing accuracy rates and pass/fail SLA status.
        """
        wer_res = await self._wer_evaluator.evaluate(
            prompt=expected_text, response=recognized_text, expected_response=expected_text
        )
        cer_res = await self._cer_evaluator.evaluate(
            prompt=expected_text, response=recognized_text, expected_response=expected_text
        )

        wer_val = float(wer_res.details.get("wer", 0.0)) if wer_res.details else 0.0
        cer_val = float(cer_res.details.get("cer", 0.0)) if cer_res.details else 0.0
        accuracy = max(0.0, 1.0 - wer_val)

        lat_results = self._latency_evaluator.evaluate_latencies(
            stt_latency=stt_latency_sec,
            llm_latency=0.0,  # Unused for strict voice-only SLA check
            tts_latency=tts_latency_sec,
            total_latency=total_latency_sec,
        )

        # Check if WER, CER, and STT/TTS/Total latency metrics passed
        lat_passed = all(m.passed for m in lat_results if m.metric_name != "LLM Latency SLA")
        passed = wer_res.passed and cer_res.passed and lat_passed
        status = MetricStatusEnum.PASS if passed else MetricStatusEnum.FAIL

        reasons = [wer_res.reason, cer_res.reason] + [m.reason for m in lat_results if not m.passed]
        explanation = " | ".join(reasons)

        logger.info(
            f"Voice Quality Evaluated | WER: {wer_val:.3f} | CER: {cer_val:.3f} | Accuracy: {accuracy*100:.1f}% | Status: {status.value}"
        )

        return VoiceQualityResult(
            wer=wer_val,
            cer=cer_val,
            recognition_accuracy=accuracy,
            stt_latency_sec=stt_latency_sec,
            tts_latency_sec=tts_latency_sec,
            total_latency_sec=total_latency_sec,
            status=status,
            explanation=explanation,
        )

    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        """
        Evaluate an end-to-end voice chatbot execution result.

        Args:
            execution_result: Target scenario execution artifacts.

        Returns:
            EvaluationResult containing all voice quality and latency SLA metrics.
        """
        start_time = time.perf_counter()
        logger.info(f"Running '{self.name}' on conversation session '{execution_result.conversation_id}'")

        expected_text = str(
            execution_result.metadata.get("expected_user_text")
            or execution_result.metadata.get("expected_text")
            or execution_result.recognized_text
        )

        wer_res = await self._wer_evaluator.evaluate(
            prompt=expected_text,
            response=execution_result.recognized_text,
            expected_response=expected_text,
        )
        cer_res = await self._cer_evaluator.evaluate(
            prompt=expected_text,
            response=execution_result.recognized_text,
            expected_response=expected_text,
        )
        latency_results = self._latency_evaluator.evaluate_latencies(
            stt_latency=execution_result.speech_latency,
            llm_latency=execution_result.llm_latency,
            tts_latency=execution_result.tts_latency,
            total_latency=execution_result.total_latency,
        )

        metric_results: List[MetricResult] = [wer_res, cer_res] + latency_results
        passed = all(m.passed for m in metric_results)
        overall_score = sum(m.score for m in metric_results) / len(metric_results) if metric_results else 0.0

        warnings: List[str] = []
        recommendations: List[str] = []
        for m in metric_results:
            if not m.passed:
                warnings.append(f"Metric '{m.metric_name}' breached SLA threshold: {m.reason}")
                recommendations.append(f"Investigate pipeline bottlenecks or audio quality for '{m.metric_name}'.")

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"'{self.name}' Finished | Overall Score: {overall_score:.2f} | Passed: {passed} | Duration: {execution_time_ms:.2f}ms"
        )

        return EvaluationResult(
            evaluator_name=self.name,
            overall_score=overall_score,
            passed=passed,
            metric_results=metric_results,
            warnings=warnings,
            recommendations=recommendations,
            execution_time_ms=execution_time_ms,
        )
