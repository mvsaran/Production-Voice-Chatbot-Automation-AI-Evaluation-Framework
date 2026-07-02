"""
Unit tests for error resilience and fallback paths in EvaluationRunner.
"""

from unittest.mock import AsyncMock
import pytest

from app.utils.exceptions import LLMException, STTException, TTSException
from evaluation.runners.evaluation_runner import EvaluationRunner


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.negative
async def test_evaluation_runner_stt_failure() -> None:
    """Test EvaluationRunner handling speech-to-text failure and using fallback prompt."""
    mock_stt = AsyncMock()
    mock_stt.speech_to_text.side_effect = STTException("Whisper service down")
    mock_llm = AsyncMock()
    mock_llm.generate_response.return_value = "Hello after fallback"
    mock_tts = AsyncMock()
    mock_tts.text_to_speech.return_value = "output.wav"
    mock_conv = AsyncMock()

    runner = EvaluationRunner(stt_service=mock_stt, llm_service=mock_llm, tts_service=mock_tts, conversation_manager=mock_conv)
    scenario_data = {
        "conversation_id": "ERR_01",
        "scenario_name": "STT Fail Test",
        "conversation": [{"user": "Fallback user text", "audio_path": "in.wav"}],
    }

    res = await runner.execute_scenario(scenario_data)
    assert res.execution_result.recognized_text == "Fallback user text"
    assert "Whisper service down" in str(res.execution_result.metadata.get("stt_error", ""))


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.negative
async def test_evaluation_runner_llm_and_tts_failure() -> None:
    """Test EvaluationRunner handling LLM and TTS failures."""
    mock_stt = AsyncMock()
    mock_stt.speech_to_text.return_value = "Hello chatbot"
    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = LLMException("OpenAI timeout")
    mock_tts = AsyncMock()
    mock_tts.text_to_speech.side_effect = TTSException("TTS engine failed")
    mock_conv = AsyncMock()

    runner = EvaluationRunner(stt_service=mock_stt, llm_service=mock_llm, tts_service=mock_tts, conversation_manager=mock_conv)
    scenario_data = {
        "conversation_id": "ERR_02",
        "scenario_name": "LLM/TTS Fail Test",
        "conversation": [{"user": "Hello", "audio_path": "in.wav"}],
    }

    res = await runner.execute_scenario(scenario_data)
    assert "Error generating response" in res.execution_result.assistant_response
    assert "OpenAI timeout" in str(res.execution_result.metadata.get("llm_error", ""))
    assert "TTS engine failed" in str(res.execution_result.metadata.get("tts_error", ""))
