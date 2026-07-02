"""
Integration tests for OpenAI LLM Service with mocked external chat completion API.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from app.config.settings import settings
from app.models.base import ConversationMessage, RoleEnum
from app.services.openai_service import OpenAILLMService
from app.utils.exceptions import LLMException


@pytest.fixture
def mock_openai_client() -> MagicMock:
    """Fixture providing a mock AsyncOpenAI client with chat completions endpoint."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.mark.asyncio
@pytest.mark.functional
async def test_llm_service_generate_response_success(mock_openai_client: MagicMock) -> None:
    """Test OpenAILLMService generating response from single user turn."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello! How can I assist with your flight booking today?"
    mock_res = MagicMock()
    mock_res.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_res

    service = OpenAILLMService(client=mock_openai_client)

    reply = await service.generate_response(
        user_message="I want to book a flight",
        conversation_history=[],
        system_prompt="You are a helpful travel assistant.",
    )

    assert reply == "Hello! How can I assist with your flight booking today?"
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful travel assistant."},
            {"role": "user", "content": "I want to book a flight"},
        ],
        temperature=0.7,
    )


@pytest.mark.asyncio
@pytest.mark.functional
async def test_llm_service_with_history(mock_openai_client: MagicMock) -> None:
    """Test OpenAILLMService formatting multi-turn conversation history correctly."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Sure, leaving on Friday from JFK."
    mock_res = MagicMock()
    mock_res.choices = [mock_choice]
    mock_openai_client.chat.completions.create.return_value = mock_res

    service = OpenAILLMService(client=mock_openai_client)
    history = [
        ConversationMessage(role=RoleEnum.USER, content="Hello"),
        ConversationMessage(role=RoleEnum.ASSISTANT, content="Hi there!"),
    ]

    await service.generate_response(
        user_message="Need flight from JFK",
        conversation_history=history,
        system_prompt="System instructions",
    )

    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=settings.MODEL_NAME,
        messages=[
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Need flight from JFK"},
        ],
        temperature=0.7,
    )


@pytest.mark.asyncio
@pytest.mark.functional
@pytest.mark.negative
async def test_llm_service_api_failure(mock_openai_client: MagicMock) -> None:
    """Test that OpenAI chat completion exception raises custom LLMException."""
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("OpenAI rate limit exceeded 429")
    service = OpenAILLMService(client=mock_openai_client)

    with pytest.raises(LLMException, match="OpenAI LLM generation failed"):
        await service.generate_response("Test prompt", [], "System")
