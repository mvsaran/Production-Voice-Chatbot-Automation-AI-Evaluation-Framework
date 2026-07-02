"""
Production In-Memory Conversation Service Implementation.

Implements BaseConversationManager interface for fast, structured conversation state
tracking, timestamp updates, and chronological turn management.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from loguru import logger

from app.config.settings import settings
from app.conversation.base import BaseConversationManager
from app.models.base import ConversationHistory, ConversationMessage, RoleEnum
from app.utils.exceptions import ConversationNotFoundException


class InMemoryConversationService(BaseConversationManager):
    """
    Production in-memory session manager for voice chatbot conversations.
    """

    def __init__(self) -> None:
        """Initialize the in-memory conversation storage dictionary."""
        self._store: Dict[str, ConversationHistory] = {}
        logger.info("Initialized InMemoryConversationService storage.")

    async def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ConversationHistory:
        """
        Create and initialize a new conversation session.

        Args:
            conversation_id: Optional custom UUID string.
            system_prompt: Optional custom system instructions.

        Returns:
            The newly created ConversationHistory object.
        """
        cid = conversation_id or f"conv_{uuid.uuid4().hex}"
        prompt = system_prompt or settings.DEFAULT_SYSTEM_PROMPT

        now = datetime.now(timezone.utc)
        history = ConversationHistory(
            conversation_id=cid,
            created_at=now,
            updated_at=now,
            messages=[],
            system_prompt=prompt,
        )
        self._store[cid] = history
        logger.info(f"Created new conversation session: '{cid}'")
        return history

    async def get_conversation(self, conversation_id: str) -> ConversationHistory:
        """
        Retrieve a conversation session by its ID.

        Args:
            conversation_id: Target session UUID.

        Returns:
            Matching ConversationHistory object.

        Raises:
            ConversationNotFoundException: If session ID is not in storage.
        """
        if conversation_id not in self._store:
            err_msg = f"Conversation session '{conversation_id}' was not found."
            logger.error(err_msg)
            raise ConversationNotFoundException(err_msg)
        return self._store[conversation_id]

    async def add_message(
        self,
        conversation_id: str,
        role: RoleEnum,
        content: str,
    ) -> ConversationMessage:
        """
        Append a new message turn to the session history and update timestamp.

        Args:
            conversation_id: Target session UUID.
            role: Speaker role (user, assistant, system).
            content: Message text content.

        Returns:
            Created ConversationMessage object.

        Raises:
            ConversationNotFoundException: If session ID is not in storage.
        """
        history = await self.get_conversation(conversation_id)
        msg = ConversationMessage(role=role, content=content)
        history.messages.append(msg)
        history.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Added [{role.value}] turn to session '{conversation_id}' | {len(content)} chars")
        return msg

    async def get_history(self, conversation_id: str) -> List[ConversationMessage]:
        """
        Retrieve chronological list of exchanged messages for a session.

        Args:
            conversation_id: Target session UUID.

        Returns:
            List of ConversationMessage items.

        Raises:
            ConversationNotFoundException: If session ID is not in storage.
        """
        history = await self.get_conversation(conversation_id)
        return history.messages

    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear all messages from a conversation session while keeping session open.

        Args:
            conversation_id: Target session UUID.

        Returns:
            True if session existed and was cleared, False otherwise.
        """
        if conversation_id not in self._store:
            logger.warning(f"Attempted to clear non-existent session: '{conversation_id}'")
            return False
        self._store[conversation_id].messages.clear()
        self._store[conversation_id].updated_at = datetime.now(timezone.utc)
        logger.info(f"Cleared all message history for session: '{conversation_id}'")
        return True
