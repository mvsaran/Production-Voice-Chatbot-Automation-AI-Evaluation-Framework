"""
Unit tests for WER, CER, Latency SLAs, and VoiceQualityEvaluator engine.
"""

from pathlib import Path
import pytest

from evaluation.engines.voice_quality import VoiceQualityEvaluator
from evaluation.metrics.cer import CEREvaluator
from evaluation.metrics.latency import LatencyEvaluator
from evaluation.metrics.wer import WEREvaluator
from evaluation.models import ExecutionResult, MetricStatusEnum


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.smoke
async def test_wer_evaluator_perfect_match() -> None:
    """Test WEREvaluator computes 0.0 WER for identical reference and hypothesis."""
    evaluator = WEREvaluator(threshold=0.15)
    res = await evaluator.evaluate(
        prompt="Book a flight to Paris",
        response="Book a flight to Paris",
        expected_response="Book a flight to Paris",
    )
    assert res.metric_name == "Word Error Rate (WER)"
    assert res.score == 1.0  # Accuracy
    assert res.details is not None and res.details["wer"] == 0.0
    assert res.passed is True
    assert res.status == MetricStatusEnum.PASS


@pytest.mark.asyncio
@pytest.mark.unit
async def test_wer_evaluator_mismatch() -> None:
    """Test WEREvaluator detects word errors and fails when above threshold."""
    evaluator = WEREvaluator(threshold=0.10)
    res = await evaluator.evaluate(
        prompt="Book a flight to Paris tomorrow",
        response="Book a train to London today",
        expected_response="Book a flight to Paris tomorrow",
    )
    assert res.details is not None and res.details["wer"] > 0.10
    assert res.passed is False
    assert res.status == MetricStatusEnum.FAIL


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cer_evaluator_perfect_match() -> None:
    """Test CEREvaluator computes 0.0 CER for identical strings."""
    evaluator = CEREvaluator(threshold=0.05)
    res = await evaluator.evaluate(
        prompt="Hello world",
        response="Hello world",
        expected_response="Hello world",
    )
    assert res.metric_name == "Character Error Rate (CER)"
    assert res.details is not None and res.details["cer"] == 0.0
    assert res.passed is True


@pytest.mark.unit
def test_latency_evaluator_slas() -> None:
    """Test LatencyEvaluator checks STT, LLM, TTS, and Total SLA limits."""
    evaluator = LatencyEvaluator(stt_threshold=1.5, llm_threshold=3.0, tts_threshold=2.0, total_threshold=6.0)
    results = evaluator.evaluate_latencies(stt_latency=1.0, llm_latency=2.0, tts_latency=1.5, total_latency=4.5)
    assert len(results) == 4
    assert all(r.passed for r in results)

    # Test SLA breach
    breach_results = evaluator.evaluate_latencies(stt_latency=2.5, llm_latency=4.0, tts_latency=3.0, total_latency=9.5)
    assert all(not r.passed for r in breach_results)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_voice_quality_evaluator_evaluate_voice() -> None:
    """Test VoiceQualityEvaluator evaluate_voice method returning VoiceQualityResult."""
    evaluator = VoiceQualityEvaluator()
    res = await evaluator.evaluate_voice(
        expected_text="Hello chatbot",
        recognized_text="Hello chatbot",
        stt_latency_sec=0.8,
        tts_latency_sec=1.2,
        total_latency_sec=2.0,
    )
    assert res.wer == 0.0
    assert res.cer == 0.0
    assert res.recognition_accuracy == 1.0
    assert res.status == MetricStatusEnum.PASS


@pytest.mark.asyncio
@pytest.mark.unit
async def test_voice_quality_evaluator_evaluate_execution_result(tmp_path: Path) -> None:
    """Test VoiceQualityEvaluator evaluate method returning aggregated EvaluationResult."""
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()

    exec_res = ExecutionResult(
        conversation_id="VQ-TEST-001",
        scenario_name="Voice Quality Test",
        recognized_text="I want to book a ticket",
        assistant_response="Where would you like to go?",
        conversation_history=[],
        audio_input_path=str(audio_path),
        audio_output_path=str(audio_path),
        speech_latency=0.5,
        llm_latency=1.0,
        tts_latency=0.7,
        total_latency=2.2,
        metadata={"expected_user_text": "I want to book a ticket"},
    )

    evaluator = VoiceQualityEvaluator()
    eval_res = await evaluator.evaluate(exec_res)

    assert eval_res.evaluator_name == "VoiceQualityEvaluator"
    assert eval_res.passed is True
    assert eval_res.overall_score == 1.0
    assert len(eval_res.metric_results) == 6  # WER, CER, STT, LLM, TTS, Total latencies
