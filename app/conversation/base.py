"""
Abstract Base Class for Conversation Management and Persistence.

Designed to allow seamless transition from in-memory conversation storage
to SQL/NoSQL databases (e.g., PostgreSQL, Redis, MongoDB) in future phases.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.base import ConversationHistory, ConversationMessage, RoleEnum


class BaseConversationManager(ABC):
    """
    Abstract interface for managing conversation state, history, and timestamps.
    """

    @abstractmethod
    async def create_conversation(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> ConversationHistory:
        """
        Initialize a new multi-turn conversation session.

        Args:
            conversation_id: Optional custom UUID. If None, generate automatically.
            system_prompt: Optional custom system prompt. If None, use configured default.

        Returns:
            The newly initialized ConversationHistory entity.
        """
        pass

    @abstractmethod
    async def get_conversation(self, conversation_id: str) -> ConversationHistory:
        """
        Retrieve the full conversation session details by its unique ID.

        Args:
            conversation_id: Target conversation UUID.

        Returns:
            The matching ConversationHistory entity.

        Raises:
            ConversationNotFoundException: If the ID does not exist in storage.
        """
        pass

    @abstractmethod
    async def add_message(
        self,
        conversation_id: str,
        role: RoleEnum,
        content: str,
    ) -> ConversationMessage:
        """
        Append a new conversational turn to the specified conversation history.

        Args:
            conversation_id: Target conversation UUID.
            role: Role of the speaker (user or assistant).
            content: Text content of the message.

        Returns:
            The created ConversationMessage object including timestamp.

        Raises:
            ConversationNotFoundException: If the ID does not exist.
        """
        pass

    @abstractmethod
    async def get_history(self, conversation_id: str) -> List[ConversationMessage]:
        """
        Retrieve chronological message turns for an active session.

        Args:
            conversation_id: Target conversation UUID.

        Returns:
            List of ConversationMessage items in chronological order.

        Raises:
            ConversationNotFoundException: If the ID does not exist.
        """
        pass

    @abstractmethod
    async def clear_conversation(self, conversation_id: str) -> bool:
        """
        Delete or reset all messages within a conversation session.

        Args:
            conversation_id: Target conversation UUID.

        Returns:
            True if cleared successfully, False otherwise.
        """
        pass
