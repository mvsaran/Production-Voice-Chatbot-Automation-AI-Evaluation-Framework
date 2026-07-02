"""
Unit tests for production InMemoryConversationService in app/services.
"""

import pytest

from app.models.base import RoleEnum
from app.services.conversation_service import InMemoryConversationService
from app.utils.exceptions import ConversationNotFoundException


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.smoke
async def test_conversation_service_create_and_get() -> None:
    """Test creating a session and retrieving it by ID."""
    service = InMemoryConversationService()
    conv = await service.create_conversation(conversation_id="TEST_001", system_prompt="System prompt")
    assert conv.conversation_id == "TEST_001"
    assert conv.system_prompt == "System prompt"
    assert len(conv.messages) == 0

    fetched = await service.get_conversation("TEST_001")
    assert fetched == conv


@pytest.mark.asyncio
@pytest.mark.functional
async def test_conversation_service_add_message_and_history() -> None:
    """Test adding user and assistant turns and retrieving history."""
    service = InMemoryConversationService()
    await service.create_conversation(conversation_id="TEST_002")

    msg1 = await service.add_message("TEST_002", RoleEnum.USER, "Hello assistant")
    msg2 = await service.add_message("TEST_002", RoleEnum.ASSISTANT, "Hello user")

    assert msg1.role == RoleEnum.USER
    assert msg1.content == "Hello assistant"
    assert msg2.role == RoleEnum.ASSISTANT

    history = await service.get_history("TEST_002")
    assert len(history) == 2
    assert history[0].content == "Hello assistant"
    assert history[1].content == "Hello user"


@pytest.mark.asyncio
@pytest.mark.functional
async def test_conversation_service_clear() -> None:
    """Test clearing message history without deleting session."""
    service = InMemoryConversationService()
    await service.create_conversation(conversation_id="TEST_003")
    await service.add_message("TEST_003", RoleEnum.USER, "Hello")

    assert len(await service.get_history("TEST_003")) == 1
    res = await service.clear_conversation("TEST_003")
    assert res is True
    assert len(await service.get_history("TEST_003")) == 0

    # Clearing non-existent conversation returns False
    assert await service.clear_conversation("NON_EXISTENT") is False


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_conversation_service_not_found() -> None:
    """Test retrieving or modifying non-existent session raises ConversationNotFoundException."""
    service = InMemoryConversationService()

    with pytest.raises(ConversationNotFoundException, match="was not found"):
        await service.get_conversation("MISSING")

    with pytest.raises(ConversationNotFoundException, match="was not found"):
        await service.add_message("MISSING", RoleEnum.USER, "Hello")

    with pytest.raises(ConversationNotFoundException, match="was not found"):
        await service.get_history("MISSING")
