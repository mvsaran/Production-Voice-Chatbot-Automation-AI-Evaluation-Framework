"""
Production Language Model Service Implementation using OpenAI & Azure OpenAI.

Implements BaseLLMService interface with dynamic endpoint routing (Standard vs Azure),
dialogue history formatting, system prompt injection, and custom exception mapping.
"""

import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI

from app.config.settings import settings
from app.llm.base import BaseLLMService
from app.models.base import ConversationMessage
from app.utils.exceptions import AuthenticationException, LLMException, NetworkTimeoutException


class OpenAILLMService(BaseLLMService):
    """
    Production LLM service routing between standard OpenAI and Azure OpenAI endpoints.
    """

    def __init__(self, client: Optional[AsyncOpenAI | AsyncAzureOpenAI] = None) -> None:
        """
        Initialize the LLM Service.

        Args:
            client: Optional injected OpenAI/AzureOpenAI client for testing and simulation.
        """
        if client:
            self._client = client
        elif settings.AZURE_OPENAI_ENABLED:
            logger.info("Initializing OpenAILLMService with Azure OpenAI endpoint.")
            self._client = AsyncAzureOpenAI(
                api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value() if settings.AZURE_OPENAI_API_KEY else "",
                api_version=settings.AZURE_OPENAI_API_VERSION,
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT or "https://mock.openai.azure.com/",
            )
        else:
            logger.info("Initializing OpenAILLMService with standard OpenAI endpoint.")
            self._client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY.get_secret_value()
            )

        self._model_name = (
            settings.AZURE_OPENAI_DEPLOYMENT_NAME if settings.AZURE_OPENAI_ENABLED else settings.MODEL_NAME
        )

    async def generate_response(
        self,
        user_message: str,
        conversation_history: List[ConversationMessage | Any],
        system_prompt: str,
    ) -> str:
        """
        Generate an intelligent conversational response from OpenAI / Azure GPT-4o.

        Args:
            user_message: Latest transcribed user text.
            conversation_history: Chronological dialogue turn items.
            system_prompt: System-level instruction guidelines.

        Returns:
            Generated text response.

        Raises:
            AuthenticationException: If API authentication fails.
            NetworkTimeoutException: If API call times out.
            LLMException: For general generation failures.
        """
        logger.debug(f"Generating LLM response using model '{self._model_name}' for input: '{user_message}'")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in conversation_history:
            role_str = turn.role.value if hasattr(turn, "role") and hasattr(turn.role, "value") else str(getattr(turn, "role", "user"))
            content_str = getattr(turn, "content", str(turn))
            messages.append({"role": role_str, "content": content_str})

        messages.append({"role": "user", "content": user_message})

        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.7,
            )
            choice = response.choices[0]
            answer: str = str(choice.message.content or "").strip()
            logger.info(f"LLM response generated successfully | Length: {len(answer)} chars")
            return answer
        except openai.AuthenticationError as e:
            err_msg = f"OpenAI API authentication failed: {str(e)}"
            logger.error(err_msg)
            raise AuthenticationException(err_msg) from e
        except (openai.APITimeoutError, asyncio.TimeoutError) as e:
            err_msg = f"OpenAI API request timed out: {str(e)}"
            logger.error(err_msg)
            raise NetworkTimeoutException(err_msg) from e
        except Exception as e:
            err_msg = f"OpenAI LLM generation failed: {str(e)}"
            logger.error(err_msg)
            raise LLMException(err_msg) from e
