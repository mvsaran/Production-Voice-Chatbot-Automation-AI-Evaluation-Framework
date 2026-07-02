"""
Unit tests for production OpenAILLMService covering initialization and exception handling.
"""

from unittest.mock import AsyncMock, patch
import pytest
from openai import APITimeoutError, AuthenticationError

from app.services.openai_service import OpenAILLMService
from app.utils.exceptions import AuthenticationException, NetworkTimeoutException


@pytest.mark.asyncio
@pytest.mark.unit
async def test_openai_service_init_standard() -> None:
    """Test initializing OpenAILLMService with standard OpenAI client."""
    with patch("app.services.openai_service.settings") as mock_settings:
        mock_settings.AZURE_OPENAI_ENABLED = False
        mock_settings.OPENAI_API_KEY.get_secret_value.return_value = "sk-test"
        mock_settings.MODEL_NAME = "gpt-4"

        service = OpenAILLMService()
        assert service._model_name == "gpt-4"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_openai_service_init_azure() -> None:
    """Test initializing OpenAILLMService with Azure OpenAI client."""
    with patch("app.services.openai_service.settings") as mock_settings:
        mock_settings.AZURE_OPENAI_ENABLED = True
        mock_settings.AZURE_OPENAI_API_KEY.get_secret_value.return_value = "azure-key"
        mock_settings.AZURE_OPENAI_API_VERSION = "2024-02-01"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://mock.azure.com"
        mock_settings.AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4-azure"

        service = OpenAILLMService()
        assert service._model_name == "gpt-4-azure"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.negative
async def test_openai_service_exceptions() -> None:
    """Test authentication and timeout error translations in generate_response."""
    mock_client = AsyncMock()
    mock_client.chat.completions.create.side_effect = AuthenticationError(
        message="Invalid key", response=AsyncMock(), body=None
    )

    service = OpenAILLMService(client=mock_client)
    with pytest.raises(AuthenticationException, match="authentication failed"):
        await service.generate_response("hello", [], "")

    mock_client.chat.completions.create.side_effect = APITimeoutError(request=AsyncMock())
    with pytest.raises(NetworkTimeoutException, match="request timed out"):
        await service.generate_response("hello", [], "")
