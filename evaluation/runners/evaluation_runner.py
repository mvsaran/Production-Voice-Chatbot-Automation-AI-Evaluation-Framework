"""
Evaluation Runner and Scenario Execution Engine.

Responsible for driving multi-turn or single-turn voice chatbot pipelines
(Speech -> Text -> LLM -> Speech), collecting execution artifacts, invoking all
registered evaluators, and aggregating final QA summary reports.

Strictly adheres to Dependency Injection, SOLID principles, and clean separation
of concerns via specialized stage executors and service resolution factories.
"""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from evaluation.adapters.mock_services import ServiceResolver
from evaluation.aggregators.qa_aggregator import EvaluationAggregator
from evaluation.config import eval_settings
from evaluation.models import AggregatedEvaluationReport, EvaluationResult, ExecutionResult
from evaluation.interfaces.evaluator import Evaluator
from evaluation.runners.scenario_loader import ScenarioLoader
from app.conversation.base import BaseConversationManager
from app.llm.base import BaseLLMService
from app.models.base import RoleEnum
from app.speech.base import BaseSpeechToTextService, BaseTextToSpeechService
from app.utils.exceptions import (
    AudioException,
    LLMException,
    STTException,
    TTSException,
    VoiceAutomationException,
)


class EvaluationRunner:
    """
    Production execution engine for automated voice chatbot QA testing.
    Uses ServiceResolver for clean dependency injection and modular stage execution.
    """

    def __init__(
        self,
        stt_service: Optional[BaseSpeechToTextService] = None,
        llm_service: Optional[BaseLLMService] = None,
        tts_service: Optional[BaseTextToSpeechService] = None,
        conversation_manager: Optional[BaseConversationManager] = None,
    ) -> None:
        """
        Initialize EvaluationRunner with dependency-injected voice chatbot services.
        If any service is omitted, ServiceResolver cleanly injects fallback adapters.
        """
        self._stt_service = ServiceResolver.resolve_stt(stt_service)
        self._llm_service = ServiceResolver.resolve_llm(llm_service)
        self._tts_service = ServiceResolver.resolve_tts(tts_service)
        self._conv_manager = ServiceResolver.resolve_conversation_manager(conversation_manager)
        self._evaluators: List[Evaluator] = []

    def register_evaluator(self, evaluator: Evaluator) -> None:
        """Register an evaluator plugin without coupling to implementation details."""
        logger.info(f"Registering QA evaluator engine: '{evaluator.name}'")
        self._evaluators.append(evaluator)

    def get_registered_evaluators(self) -> List[Evaluator]:
        """Return list of all currently registered evaluators."""
        return list(self._evaluators)

    def load_scenario(self, file_path: Path | str) -> Dict[str, Any]:
        """Delegate scenario file loading and validation to ScenarioLoader."""
        return ScenarioLoader.load_scenario(file_path)

    async def execute_scenario(self, scenario_data: Dict[str, Any]) -> AggregatedEvaluationReport:
        """
        Execute an automated voice chatbot scenario end-to-end and invoke evaluators.

        Args:
            scenario_data: Loaded scenario dictionary.

        Returns:
            AggregatedEvaluationReport merging results across all registered evaluators.
        """
        scenario_id = scenario_data.get("conversation_id", str(uuid.uuid4()))
        scenario_name = scenario_data.get("scenario_name", f"Scenario-{scenario_id}")
        system_prompt = scenario_data.get("system_prompt", eval_settings.get_openai_api_key_str() and "Default System Prompt")
        turns: List[Dict[str, str]] = scenario_data.get("conversation", [])
        expected_behavior: str = str(scenario_data.get("expected_behavior", ""))

        # Bind contextual logger for automated trace correlation
        ctx_logger = logger.bind(scenario_id=scenario_id, scenario_name=scenario_name)
        ctx_logger.info(f"Scenario Started: '{scenario_name}' (ID: {scenario_id})")
        total_start = time.perf_counter()

        # 1. Start Conversation Session
        ctx_logger.info(f"Conversation Started: {scenario_id}")
        await self._conv_manager.create_conversation(conversation_id=scenario_id, system_prompt=system_prompt)

        last_turn = turns[-1] if turns else {"user": "Hello", "audio_path": "test_data/sample.wav"}
        user_input_text = last_turn.get("user", "")
        audio_input_path = last_turn.get("audio_path", "test_data/sample.wav")

        # 2. Speech To Text (STT) Stage
        recognized_text, stt_latency_sec, stt_err = await self._execute_stt_stage(audio_input_path, user_input_text, ctx_logger)

        await self._conv_manager.add_message(scenario_id, RoleEnum.USER, recognized_text)
        history = await self._conv_manager.get_history(scenario_id)

        # 3. LLM Response Stage
        assistant_response, llm_latency_sec, llm_err = await self._execute_llm_stage(recognized_text, history, system_prompt, ctx_logger)

        await self._conv_manager.add_message(scenario_id, RoleEnum.ASSISTANT, assistant_response)
        updated_history = await self._conv_manager.get_history(scenario_id)
        formatted_history = [
            {"role": msg.role.value if hasattr(msg, "role") else "user", "content": getattr(msg, "content", str(msg))}
            for msg in updated_history
        ]

        # 4. Text To Speech (TTS) Stage
        audio_output_path, tts_latency_sec, tts_err = await self._execute_tts_stage(assistant_response, scenario_id, ctx_logger)

        total_latency_sec = time.perf_counter() - total_start

        # 5. Collect Execution Artifacts -> Create ExecutionResult object
        metadata = {
            "expected_behavior": expected_behavior,
            "expected_answer": last_turn.get("expected", expected_behavior or assistant_response),
            "expected_user_text": user_input_text,
            "stt_error": stt_err,
            "llm_error": llm_err,
            "tts_error": tts_err,
        }
        execution_result = ExecutionResult(
            conversation_id=scenario_id,
            scenario_name=scenario_name,
            recognized_text=recognized_text,
            assistant_response=assistant_response,
            conversation_history=formatted_history,
            audio_input_path=str(audio_input_path),
            audio_output_path=audio_output_path,
            speech_latency=stt_latency_sec,
            llm_latency=llm_latency_sec,
            tts_latency=tts_latency_sec,
            total_latency=total_latency_sec,
            metadata=metadata,
        )

        # 6. Invoke All Registered Evaluators
        evaluator_results = await self._invoke_evaluators(execution_result, ctx_logger)

        # 7. Aggregate Results & Generate Summary
        report = EvaluationAggregator.aggregate(execution_result, evaluator_results)
        ctx_logger.info(
            f"Scenario Completed: '{scenario_name}' | Overall QA Score: {report.overall_score:.2f} | Execution Time: {total_latency_sec:.2f}s"
        )
        return report

    async def _execute_stt_stage(
        self, audio_path: str, fallback_text: str, ctx_logger: Any
    ) -> Tuple[str, float, Optional[str]]:
        """Execute Speech-to-Text stage with error resilience."""
        stt_start = time.perf_counter()
        recognized_text = fallback_text
        stt_err = None
        try:
            ctx_logger.debug(f"Executing Speech-to-Text on audio: {audio_path}")
            recognized_text = await self._stt_service.speech_to_text(audio_path)
            ctx_logger.info(f"Speech Completed | Recognized: '{recognized_text}'")
        except (STTException, AudioException, Exception) as e:
            stt_err = f"Speech recognition failure: {str(e)}"
            ctx_logger.error(stt_err)
            recognized_text = fallback_text or "Speech recognition failed"
        return recognized_text, time.perf_counter() - stt_start, stt_err

    async def _execute_llm_stage(
        self, prompt_text: str, history: List[Any], system_prompt: str, ctx_logger: Any
    ) -> Tuple[str, float, Optional[str]]:
        """Execute LLM response generation stage with error resilience."""
        llm_start = time.perf_counter()
        assistant_response = ""
        llm_err = None
        try:
            ctx_logger.debug(f"Sending prompt to LLM: '{prompt_text}'")
            assistant_response = await self._llm_service.generate_response(prompt_text, history, system_prompt)
            ctx_logger.info(f"LLM Completed | Response: '{assistant_response[:60]}...'")
        except (LLMException, VoiceAutomationException, Exception) as e:
            llm_err = f"LLM generation failure: {str(e)}"
            ctx_logger.error(llm_err)
            assistant_response = f"Error generating response: {llm_err}"
        return assistant_response, time.perf_counter() - llm_start, llm_err

    async def _execute_tts_stage(
        self, text: str, scenario_id: str, ctx_logger: Any
    ) -> Tuple[str, float, Optional[str]]:
        """Execute Text-to-Speech synthesis stage with error resilience."""
        tts_start = time.perf_counter()
        audio_output_path = f"audio/output/{scenario_id}.wav"
        tts_err = None
        try:
            ctx_logger.debug("Synthesizing speech for response...")
            out_path = await self._tts_service.text_to_speech(text)
            audio_output_path = str(out_path)
            ctx_logger.info(f"TTS Completed | Audio saved to: {audio_output_path}")
        except (TTSException, VoiceAutomationException, Exception) as e:
            tts_err = f"TTS synthesis failure: {str(e)}"
            ctx_logger.error(tts_err)
        return audio_output_path, time.perf_counter() - tts_start, tts_err

    async def _invoke_evaluators(
        self, execution_result: ExecutionResult, ctx_logger: Any
    ) -> List[EvaluationResult]:
        """Invoke all registered evaluators inside error-resilient wrappers."""
        evaluator_results: List[EvaluationResult] = []
        for evaluator in self._evaluators:
            ctx_logger.info(f"{evaluator.name} Started for scenario {execution_result.conversation_id}...")
            try:
                res = await evaluator.evaluate(execution_result)
                ctx_logger.info(f"{evaluator.name} Completed | Score: {res.overall_score:.2f}")
                evaluator_results.append(res)
            except Exception as e:
                err_msg = f"Evaluator '{evaluator.name}' raised unhandled exception: {str(e)}"
                ctx_logger.error(err_msg)
                evaluator_results.append(
                    EvaluationResult(
                        evaluator_name=evaluator.name,
                        overall_score=0.0,
                        passed=False,
                        error=err_msg,
                        warnings=[err_msg],
                    )
                )
        return evaluator_results
