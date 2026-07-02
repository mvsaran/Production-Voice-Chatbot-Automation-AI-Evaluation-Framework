"""
Evaluation Aggregator and Central QA Orchestrator.

Merges results from every registered Evaluator (DeepEval, RAGAS, Voice QA, Safety),
computes overall quality composite scores, pass rates, and compiles consolidated
warnings, failed metrics, and actionable recommendations.
"""

import time
from typing import List
from loguru import logger
from evaluation.models import AggregatedEvaluationReport, EvaluationResult, ExecutionResult


class EvaluationAggregator:
    """
    Central aggregator responsible for merging multi-evaluator results into a single
    unified quality assurance report.
    """

    @classmethod
    def aggregate(
        self,
        execution_result: ExecutionResult,
        evaluator_results: List[EvaluationResult],
    ) -> AggregatedEvaluationReport:
        """
        Merge results from every registered evaluator into an AggregatedEvaluationReport.

        Args:
            execution_result: Target scenario execution artifacts.
            evaluator_results: List of individual results from all evaluators.

        Returns:
            Consolidated AggregatedEvaluationReport.
        """
        start_time = time.perf_counter()
        logger.info(f"Aggregating evaluation results from {len(evaluator_results)} evaluators...")

        if not evaluator_results:
            logger.warning("No evaluator results provided for aggregation.")
            return AggregatedEvaluationReport(
                conversation_id=execution_result.conversation_id,
                scenario_name=execution_result.scenario_name,
                execution_result=execution_result,
                overall_score=0.0,
                pass_rate=0.0,
                passed=False,
                warnings=["No evaluators registered or executed."],
            )

        total_metrics = 0
        passed_metrics = 0
        sum_scores = 0.0
        failed_metrics: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []

        for eval_res in evaluator_results:
            if eval_res.error:
                warnings.append(f"Evaluator '{eval_res.evaluator_name}' reported error: {eval_res.error}")

            warnings.extend([f"[{eval_res.evaluator_name}] {w}" for w in eval_res.warnings])
            recommendations.extend([f"[{eval_res.evaluator_name}] {r}" for r in eval_res.recommendations])

            for metric in eval_res.metric_results:
                total_metrics += 1
                if metric.passed:
                    passed_metrics += 1
                else:
                    failed_metrics.append(f"[{eval_res.evaluator_name}] {metric.metric_name} (Score: {metric.score:.2f})")

        valid_eval_scores = [e.overall_score for e in evaluator_results if e.error is None]
        overall_score = sum(valid_eval_scores) / len(valid_eval_scores) if valid_eval_scores else 0.0
        pass_rate = (passed_metrics / total_metrics) if total_metrics > 0 else (len([e for e in evaluator_results if e.passed]) / len(evaluator_results) if evaluator_results else 0.0)
        all_evaluators_passed = all(e.passed for e in evaluator_results) and (len(failed_metrics) == 0)

        total_time_ms = sum(e.execution_time_ms for e in evaluator_results) + ((time.perf_counter() - start_time) * 1000.0)

        logger.info(
            f"Aggregation finished | Overall Score: {overall_score:.2f} | Pass Rate: {pass_rate*100:.1f}% | Failed Metrics: {len(failed_metrics)}"
        )

        return AggregatedEvaluationReport(
            conversation_id=execution_result.conversation_id,
            scenario_name=execution_result.scenario_name,
            execution_result=execution_result,
            overall_score=overall_score,
            pass_rate=pass_rate,
            passed=all_evaluators_passed,
            evaluator_results=evaluator_results,
            failed_metrics=failed_metrics,
            warnings=warnings,
            recommendations=recommendations,
            total_evaluation_time_ms=total_time_ms,
        )
