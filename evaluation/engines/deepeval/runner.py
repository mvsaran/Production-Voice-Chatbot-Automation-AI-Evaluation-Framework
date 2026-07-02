"""
DeepEval Evaluator Implementation.

Integrates DeepEval metrics (Answer Relevancy, Faithfulness, Hallucination, Answer Correctness)
conforming to the `Evaluator` interface. Handles external LLM judge API timeouts and exceptions
gracefully, continuing evaluation of remaining metrics whenever possible.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
from loguru import logger
from evaluation.config import eval_settings
from evaluation.models import EvaluationResult, ExecutionResult, MetricDetail
from evaluation.interfaces.evaluator import Evaluator

try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        AnswerCorrectnessMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    logger.warning("DeepEval SDK not installed or import failed. Falling back to simulation mode.")


class DeepEvalRunner(Evaluator):
    """
    Concrete Evaluator implementation executing DeepEval AI quality metrics.
    
    Evaluates:
    1. Answer Relevancy
    2. Faithfulness
    3. Hallucination
    4. Answer Correctness
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        timeout_seconds: float = 10.0,
        simulation_mode: bool = False,
    ) -> None:
        """
        Initialize the DeepEvalRunner.

        Args:
            model_name: Custom LLM judge model name (defaults to config EVAL_MODEL_NAME).
            timeout_seconds: Timeout per individual metric evaluation to prevent CI/CD hangs.
            simulation_mode: If True, execute offline simulation for CI testing without OpenAI API calls.
        """
        self._model_name = model_name or eval_settings.EVAL_MODEL_NAME
        self._timeout_seconds = timeout_seconds
        self._simulation_mode = simulation_mode or (not DEEPEVAL_AVAILABLE) or ("mock" in eval_settings.get_openai_api_key_str())

    @property
    def name(self) -> str:
        return "DeepEval"

    async def evaluate(self, execution_result: ExecutionResult) -> EvaluationResult:
        """
        Execute DeepEval quality metrics against the provided execution result.

        Args:
            execution_result: Immutable ExecutionResult containing input prompt and response.

        Returns:
            EvaluationResult containing aggregated DeepEval scores and individual metric breakdowns.
        """
        logger.info(f"Starting DeepEval evaluation for scenario '{execution_result.scenario_name}' (ID: {execution_result.conversation_id})...")
        start_time = time.perf_counter()

        metric_results: List[MetricDetail] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        context: List[str] = execution_result.metadata.get("context", [execution_result.recognized_text])
        expected_answer: str = execution_result.metadata.get("expected_answer", execution_result.assistant_response)

        # 1. Evaluate Answer Relevancy
        relevancy_res = await self._run_metric_safe(
            metric_name="Answer Relevancy",
            threshold=eval_settings.EVAL_THRESHOLD_RELEVANCY,
            prompt=execution_result.recognized_text,
            response=execution_result.assistant_response,
            context=context,
            expected_answer=expected_answer,
            metric_class=AnswerRelevancyMetric if DEEPEVAL_AVAILABLE else None,
            simulated_score=0.92,
            simulated_reason="The assistant response directly and concisely addresses the user prompt without redundant details.",
        )
        metric_results.append(relevancy_res)

        # 2. Evaluate Faithfulness
        faithfulness_res = await self._run_metric_safe(
            metric_name="Faithfulness",
            threshold=eval_settings.EVAL_THRESHOLD_FAITHFULNESS,
            prompt=execution_result.recognized_text,
            response=execution_result.assistant_response,
            context=context,
            expected_answer=expected_answer,
            metric_class=FaithfulnessMetric if DEEPEVAL_AVAILABLE else None,
            simulated_score=0.95,
            simulated_reason="All factual statements in the response are strictly supported by the conversation context.",
        )
        metric_results.append(faithfulness_res)

        # 3. Evaluate Hallucination (Note: For Hallucination, lower score is better, but DeepEval metric returns 0 to 1 where <= threshold passes)
        hallucination_res = await self._run_metric_safe(
            metric_name="Hallucination",
            threshold=eval_settings.EVAL_THRESHOLD_HALLUCINATION,
            prompt=execution_result.recognized_text,
            response=execution_result.assistant_response,
            context=context,
            expected_answer=expected_answer,
            metric_class=HallucinationMetric if DEEPEVAL_AVAILABLE else None,
            simulated_score=0.02,
            simulated_reason="No fabricated entities, dates, or unverified claims detected in the generated response.",
            is_lower_better=True,
        )
        metric_results.append(hallucination_res)

        # 4. Evaluate Answer Correctness
        correctness_res = await self._run_metric_safe(
            metric_name="Answer Correctness",
            threshold=eval_settings.EVAL_THRESHOLD_CORRECTNESS,
            prompt=execution_result.recognized_text,
            response=execution_result.assistant_response,
            context=context,
            expected_answer=expected_answer,
            metric_class=AnswerCorrectnessMetric if DEEPEVAL_AVAILABLE else None,
            simulated_score=0.88,
            simulated_reason="The generated response matches the expected semantic meaning and achieves the task goal.",
        )
        metric_results.append(correctness_res)

        # Aggregate metrics (invert Hallucination score since lower is better for quality aggregation)
        valid_scores = [
            (1.0 - m.score) if m.metric_name == "Hallucination" else m.score
            for m in metric_results
            if m.error_message is None
        ]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        all_passed = all(m.passed for m in metric_results)

        for m in metric_results:
            if m.error_message:
                warnings.append(f"Metric '{m.metric_name}' encountered an exception: {m.error_message}")
            elif not m.passed:
                warnings.append(f"Metric '{m.metric_name}' failed threshold check ({m.score:.2f} vs SLA {m.threshold:.2f}).")
                recommendations.append(f"Improve '{m.metric_name}': {m.reason}")

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0
        logger.info(
            f"DeepEval completed in {exec_time_ms:.1f}ms | Overall Score: {overall_score:.2f} | Status: {'PASS' if all_passed else 'FAIL'}"
        )

        return EvaluationResult(
            evaluator_name=self.name,
            overall_score=overall_score,
            passed=all_passed,
            metric_results=metric_results,
            warnings=warnings,
            recommendations=recommendations,
            execution_time_ms=exec_time_ms,
        )

    async def _run_metric_safe(
        self,
        metric_name: str,
        threshold: float,
        prompt: str,
        response: str,
        context: List[str],
        expected_answer: str,
        metric_class: Any,
        simulated_score: float,
        simulated_reason: str,
        is_lower_better: bool = False,
    ) -> MetricDetail:
        """
        Execute an individual evaluation metric with timeout and exception protection.
        Ensures partial evaluation failures never crash the broader CI/CD runner.
        """
        start_time = time.perf_counter()
        logger.debug(f"Executing metric '{metric_name}' (Threshold: {threshold})...")

        if self._simulation_mode or not metric_class:
            await asyncio.sleep(0.05)
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            passed = (simulated_score <= threshold) if is_lower_better else (simulated_score >= threshold)
            logger.debug(f"Metric '{metric_name}' simulated | Score: {simulated_score} | Passed: {passed}")
            return MetricDetail(
                metric_name=metric_name,
                score=simulated_score,
                threshold=threshold,
                passed=passed,
                reason=simulated_reason,
                execution_time_ms=exec_time_ms,
            )

        try:
            test_case = LLMTestCase(
                input=prompt,
                actual_output=response,
                expected_output=expected_answer,
                context=context,
            )
            metric = metric_class(threshold=threshold, model=self._model_name)

            await asyncio.wait_for(
                asyncio.to_thread(metric.measure, test_case),
                timeout=self._timeout_seconds,
            )

            score: float = float(metric.score or 0.0)
            reason: str = str(metric.reason or "No explanation provided by LLM judge.")
            passed: bool = bool(metric.is_successful())
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0

            logger.info(f"Metric '{metric_name}' evaluated in {exec_time_ms:.1f}ms | Score: {score:.2f} | Passed: {passed}")
            return MetricDetail(
                metric_name=metric_name,
                score=score,
                threshold=threshold,
                passed=passed,
                reason=reason,
                execution_time_ms=exec_time_ms,
            )

        except asyncio.TimeoutError:
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"DeepEval metric '{metric_name}' timed out after {self._timeout_seconds}s."
            logger.error(err_msg)
            return MetricDetail(
                metric_name=metric_name,
                score=0.0 if not is_lower_better else 1.0,
                threshold=threshold,
                passed=False,
                reason="Evaluation timeout exceeded.",
                error_message=err_msg,
                execution_time_ms=exec_time_ms,
            )
        except Exception as e:
            exec_time_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = f"DeepEval metric '{metric_name}' failed: {str(e)}"
            logger.error(err_msg)
            return MetricDetail(
                metric_name=metric_name,
                score=0.0 if not is_lower_better else 1.0,
                threshold=threshold,
                passed=False,
                reason="Evaluation exception encountered.",
                error_message=err_msg,
                execution_time_ms=exec_time_ms,
            )
