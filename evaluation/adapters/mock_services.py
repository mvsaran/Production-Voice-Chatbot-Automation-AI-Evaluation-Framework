"""
Mock and Simulation Adapters for Offline CI/CD Testing and Dependency Injection.

Extracted out of core runners to adhere to the Single Responsibility Principle (SRP)
and Dependency Inversion Principle (DIP). Provides standalone simulation capabilities
when live OpenAI or Azure API endpoints are offline or unavailable.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from evaluation.config import eval_settings
from app.conversation.base import BaseConversationManager
from app.llm.base import BaseLLMService
from app.models.base import ConversationHistory, RoleEnum
from app.speech.base import BaseSpeechToTextService, BaseTextToSpeechService
from app.utils.exceptions import ConversationNotFoundException


class MockSpeechToTextAdapter(BaseSpeechToTextService):
    """Fallback simulated STT service for offline CI/CD scenario execution."""
    async def speech_to_text(self, audio_path: Path | str) -> str:
        await asyncio.sleep(0.05)
        return "Book ticket tomorrow"


class MockLLMServiceAdapter(BaseLLMService):
    """Fallback simulated LLM service for offline CI/CD scenario execution."""
    async def generate_response(self, user_message: str, conversation_history: List[Any], system_prompt: str) -> str:
        await asyncio.sleep(0.1)
        return "Sure, where is your destination and what time would you like to depart?"


class MockTextToSpeechAdapter(BaseTextToSpeechService):
    """Fallback simulated TTS service for offline CI/CD scenario execution."""
    async def text_to_speech(self, text: str, output_path: Optional[Path | str] = None) -> Path:
        await asyncio.sleep(0.05)
        out = Path(output_path or eval_settings.REPORTS_HTML_DIR / f"tts_{uuid.uuid4().hex[:6]}.wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.touch(exist_ok=True)
        return out


class InMemoryConversationAdapter(BaseConversationManager):
    """Fallback in-memory conversation manager adapter."""
    def __init__(self) -> None:
        self._store: Dict[str, ConversationHistory] = {}

    async def create_conversation(self, conversation_id: Optional[str] = None, system_prompt: Optional[str] = None) -> ConversationHistory:
        cid = conversation_id or str(uuid.uuid4())
        hist = ConversationHistory(conversation_id=cid, system_prompt=system_prompt or "Default prompt")
        self._store[cid] = hist
        return hist

    async def get_conversation(self, conversation_id: str) -> ConversationHistory:
        if conversation_id not in self._store:
            raise ConversationNotFoundException(f"Conversation {conversation_id} not found")
        return self._store[conversation_id]

    async def add_message(self, conversation_id: str, role: RoleEnum, content: str) -> Any:
        hist = await self._get_or_create(conversation_id)
        from app.models.base import ConversationMessage
        msg = ConversationMessage(role=role, content=content)
        hist.messages.append(msg)
        return msg

    async def get_history(self, conversation_id: str) -> List[Any]:
        if conversation_id not in self._store:
            return []
        return self._store[conversation_id].messages

    async def clear_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._store:
            self._store[conversation_id].messages.clear()
            return True
        return False

    async def _get_or_create(self, cid: str) -> ConversationHistory:
        if cid not in self._store:
            return await self.create_conversation(conversation_id=cid)
        return self._store[cid]


class ServiceResolver:
    """
    Dependency Injection Factory for resolving Voice Chatbot services.
    Enforces clean separation between live production services and CI simulation adapters.
    """

    @classmethod
    def resolve_stt(cls, stt_service: Optional[BaseSpeechToTextService] = None) -> BaseSpeechToTextService:
        if stt_service:
            return stt_service
        logger.debug("No STT service injected; resolving default MockSpeechToTextAdapter.")
        return MockSpeechToTextAdapter()

    @classmethod
    def resolve_llm(cls, llm_service: Optional[BaseLLMService] = None) -> BaseLLMService:
        if llm_service:
            return llm_service
        logger.debug("No LLM service injected; resolving default MockLLMServiceAdapter.")
        return MockLLMServiceAdapter()

    @classmethod
    def resolve_tts(cls, tts_service: Optional[BaseTextToSpeechService] = None) -> BaseTextToSpeechService:
        if tts_service:
            return tts_service
        logger.debug("No TTS service injected; resolving default MockTextToSpeechAdapter.")
        return MockTextToSpeechAdapter()

    @classmethod
    def resolve_conversation_manager(cls, conv_manager: Optional[BaseConversationManager] = None) -> BaseConversationManager:
        if conv_manager:
            return conv_manager
        logger.debug("No Conversation Manager injected; resolving default InMemoryConversationAdapter.")
        return InMemoryConversationAdapter()
