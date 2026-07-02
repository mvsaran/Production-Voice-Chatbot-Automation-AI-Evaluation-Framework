"""
Live Interactive Chatbot Test Script.

This standalone script instantiates the core voice chatbot automation services
(Conversation Service, OpenAI LLM Service, and TTS Service) and executes a live
end-to-end conversation turn using your configured OpenAI API key.
"""

import asyncio
import sys
import time
from app.config.settings import settings
from app.models.base import RoleEnum
from app.services import InMemoryConversationService, OpenAILLMService, OpenAITTSService
from loguru import logger

# Ensure stdout uses UTF-8 or fall back safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def main():
    print("=" * 70)
    print("  LIVE INTERACTIVE VOICE CHATBOT TEST ENGINE")
    print("=" * 70)

    # 1. Initialize Core Architecture Services
    print("\n[1/3] Initializing Core Architecture Services...")
    conv_service = InMemoryConversationService()
    llm_service = OpenAILLMService()
    tts_service = OpenAITTSService()
    print("      [OK] Services successfully loaded via Dependency Injection.")

    # 2. Define Live Test Session and Prompt
    session_id = "LIVE_TEST_SESSION_001"
    user_prompt = (
        "Hello! I am testing the live voice chatbot engine. "
        "Can you check my primary checking account balance and tell me my last transaction?"
    )

    print(f"\n[2/3] Sending Turn to LLM & Conversation Orchestrator...")
    print(f"      User Prompt : \"{user_prompt}\"")
    print("      Waiting for OpenAI GPT-4o reasoning response...")

    # 3. Process Turn End-to-End
    try:
        start_time = time.perf_counter()
        
        # Create session in memory
        await conv_service.create_conversation(
            conversation_id=session_id,
            system_prompt="You are a helpful AI banking assistant. Give concise and professional answers."
        )
        
        # Add user turn to history
        await conv_service.add_message(session_id, RoleEnum.USER, user_prompt)
        history = await conv_service.get_history(session_id)

        # Generate response from OpenAI LLM Service
        assistant_response = await llm_service.generate_response(
            user_message=user_prompt,
            conversation_history=history,
            system_prompt="You are a helpful AI banking assistant. Give concise and professional answers."
        )
        llm_latency = time.perf_counter() - start_time

        # Store assistant response in conversation state
        await conv_service.add_message(session_id, RoleEnum.ASSISTANT, assistant_response)
        updated_history = await conv_service.get_history(session_id)

        print("\n[3/3] Execution Complete!")
        print("=" * 70)
        print(f"ASSISTANT RESPONSE:\n{assistant_response}")
        print("=" * 70)
        print("\nTELEMETRY & PERFORMANCE METRICS:")
        print(f"   * Session ID        : {session_id}")
        print(f"   * LLM Latency       : {llm_latency:.2f} seconds")
        print(f"   * Turn History Count: {len(updated_history)} messages stored in session")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Test script execution failed: {e}")
        print(f"\nERROR: Failed to execute turn: {e}")


if __name__ == "__main__":
    asyncio.run(main())
