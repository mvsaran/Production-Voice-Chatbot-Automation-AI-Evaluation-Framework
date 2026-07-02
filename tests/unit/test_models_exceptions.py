"""
Unit tests for domain models, Pydantic validation, and custom exception hierarchy.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError

from app.models.base import (
    AudioMetadata,
    ConversationHistory,
    ConversationMessage,
    PipelineLatency,
    RoleEnum,
)
from app.utils.exceptions import (
    AggregationException,
    AuthenticationException,
    ConfigurationException,
    ConversationNotFoundException,
    EmptyAudioException,
    EvaluationException,
    EvaluatorExecutionException,
    LLMException,
    MetricCalculationException,
    NetworkTimeoutException,
    ScenarioLoadException,
    ScenarioValidationException,
    STTException,
    TTSException,
    VoiceAutomationException,
)
from evaluation.models import (
    AggregatedEvaluationReport,
    ConversationEvaluationReport,
    ConversationScenario,
    ConversationValidationResult,
    DashboardReadyEvaluationOutput,
    EvaluationResult,
    ExecutionResult,
    MetricDetail,
    MetricResult,
    MetricStatusEnum,
    ScenarioTurn,
    SecurityEvaluationResult,
    VoiceQualityResult,
)


@pytest.mark.smoke
@pytest.mark.functional
def test_all_custom_exceptions_initialization_and_str() -> None:
    """Test instantiation and string representation of all custom exceptions."""
    base_exc = VoiceAutomationException("Base error", details="code: 500")
    assert "Base error" in str(base_exc)
    assert base_exc.details == "code: 500"

    exceptions = [
        ConfigurationException("Config failed"),
        EmptyAudioException("Empty audio"),
        STTException("STT failed"),
        TTSException("TTS failed"),
        AuthenticationException("Auth failed"),
        NetworkTimeoutException("Timeout"),
        LLMException("LLM failed"),
        ConversationNotFoundException("Not found"),
        EvaluationException("Eval failed"),
        ScenarioLoadException("Scenario load failed"),
        ScenarioValidationException("Scenario invalid"),
        EvaluatorExecutionException("Evaluator failed"),
        MetricCalculationException("Metric calc failed"),
        AggregationException("Aggregation failed"),
    ]

    for exc in exceptions:
        assert isinstance(exc, VoiceAutomationException)
        assert len(str(exc)) > 0


@pytest.mark.functional
def test_app_domain_models_validation() -> None:
    """Test creation and validation of core app Pydantic models."""
    msg = ConversationMessage(role=RoleEnum.USER, content="Hello voice bot")
    assert msg.role == RoleEnum.USER
    assert msg.content == "Hello voice bot"
    assert isinstance(msg.timestamp, datetime)

    with pytest.raises(ValidationError):
        msg.content = "New content"  # type: ignore

    now = datetime.now(timezone.utc)
    hist = ConversationHistory(
        conversation_id="conv_123",
        created_at=now,
        updated_at=now,
        messages=[msg],
        system_prompt="Be concise",
    )
    assert hist.conversation_id == "conv_123"
    assert len(hist.messages) == 1

    audio_meta = AudioMetadata(
        file_path=Path("audio/test.wav"),
        format="wav",
        duration_seconds=2.5,
        file_size_bytes=1024,
    )
    assert audio_meta.format == "wav"

    latency = PipelineLatency(stt_latency_ms=100.0, llm_latency_ms=300.0, tts_latency_ms=200.0)
    assert latency.stt_latency_ms == 100.0


@pytest.mark.functional
def test_evaluation_scenarios_models() -> None:
    """Test Pydantic validation for scenario definitions."""
    turn = ScenarioTurn(turn_id=1, user_prompt="hi", input_audio_path="test.wav", expected_answer="greeting")
    assert turn.turn_id == 1

    scenario = ConversationScenario(
        scenario_id="SCEN_001",
        scenario_name="Test scenario",
        description="Test description",
        turns=[turn],
    )
    assert scenario.scenario_id == "SCEN_001"
    assert scenario.metadata == {}


@pytest.mark.functional
def test_evaluation_results_models() -> None:
    """Test execution result and metric detail models."""
    exec_res = ExecutionResult(
        conversation_id="EXEC_001",
        scenario_name="Test Exec",
        recognized_text="Hello",
        assistant_response="Hi there",
        conversation_history=[],
        audio_input_path="in.wav",
        audio_output_path="out.wav",
    )
    assert exec_res.speech_latency == 0.0
    assert exec_res.total_latency == 0.0

    detail = MetricDetail(metric_name="Faithfulness", score=0.9, threshold=0.7, passed=True, reason="Good")
    assert detail.passed is True

    metric_res = MetricResult(
        metric_name="Answer Relevancy",
        score=0.95,
        threshold=0.7,
        passed=True,
        details={"info": "ok"},
    )
    assert metric_res.passed is True
    assert metric_res.error_message is None

    voice_qual = VoiceQualityResult(
        wer=0.05,
        cer=0.01,
        recognition_accuracy=0.95,
        stt_latency_sec=0.1,
        tts_latency_sec=0.05,
        total_latency_sec=0.15,
        status=MetricStatusEnum.PASS,
        explanation="Good accuracy",
    )
    assert voice_qual.wer == 0.05

    conv_val = ConversationValidationResult(
        conversation_id="c1",
        history_maintained=True,
        context_preserved=True,
        follow_up_correct=True,
        intent_consistent=True,
        task_completed=True,
        loop_detected=False,
        duplicate_detected=False,
        conversation_score=1.0,
        status=MetricStatusEnum.PASS,
        explanation="Flow verified",
    )
    assert conv_val.task_completed is True

    sec_eval = SecurityEvaluationResult(
        prompt_injection_protected=True,
        pii_leakage_detected=False,
        unsafe_content_detected=False,
        security_score=1.0,
        status=MetricStatusEnum.PASS,
        explanation="Safe",
    )
    assert sec_eval.prompt_injection_protected is True

    eval_res = EvaluationResult(
        evaluator_name="DeepEval",
        overall_score=0.9,
        passed=True,
        metric_results=[metric_res],
        warnings=["Minor warning"],
        recommendations=["Keep going"],
    )
    assert eval_res.overall_score == 0.9
    assert len(eval_res.warnings) == 1


@pytest.mark.functional
def test_evaluation_report_models() -> None:
    """Test aggregated quality assurance report models."""
    report = AggregatedEvaluationReport(
        conversation_id="REP_001",
        scenario_name="Flight Booking",
        execution_result=ExecutionResult(
            conversation_id="REP_001",
            scenario_name="Flight Booking",
            recognized_text="Book flight",
            assistant_response="Booking confirmed",
            conversation_history=[],
            audio_input_path="in.wav",
            audio_output_path="out.wav",
        ),
        evaluator_results=[],
        overall_score=0.85,
        pass_rate=1.0,
        passed=True,
    )
    assert len(report.evaluator_results) == 0
    assert report.passed is True

    dashboard = DashboardReadyEvaluationOutput(
        conversation_id="REP_001",
        prompt="Book flight",
        response="Booking confirmed",
        correctness=0.9,
        status=MetricStatusEnum.PASS,
    )
    assert dashboard.status == MetricStatusEnum.PASS
    assert isinstance(dashboard, ConversationEvaluationReport)
