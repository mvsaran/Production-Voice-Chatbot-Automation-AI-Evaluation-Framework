"""
End-to-End Voice Pipeline Test Script (Speech -> LLM -> Speech).

This script demonstrates how to execute complete voice-to-voice conversational turns:
1. Synthesizes a simulated user voice prompt to WAV (or loads an existing audio file).
2. Transcribes user audio to text using OpenAI Whisper API.
3. Generates an assistant response using OpenAI GPT-4o.
4. Synthesizes the assistant response back to spoken WAV audio using OpenAI TTS.
"""

import asyncio
from pathlib import Path
import sys
import time
from app.config.settings import settings
from app.models.base import RoleEnum
from app.services import (
    InMemoryConversationService,
    OpenAILLMService,
    OpenAITTSService,
    OpenAIWhisperService,
)
from loguru import logger

# Ensure stdout uses UTF-8 or fall back safely on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def main():
    print("=" * 70)
    print("  LIVE END-TO-END VOICE CHATBOT PIPELINE (STT -> LLM -> TTS)")
    print("=" * 70)

    # 1. Initialize Voice Architecture Services
    print("\n[1/4] Initializing Voice Services (Whisper, GPT-4o, OpenAI TTS)...")
    conv_service = InMemoryConversationService()
    stt_service = OpenAIWhisperService()
    llm_service = OpenAILLMService()
    tts_service = OpenAITTSService()
    print("      [OK] All speech and language services initialized.")

    session_id = "VOICE_TURN_SESSION_001"
    await conv_service.create_conversation(
        conversation_id=session_id,
        system_prompt="You are a friendly AI customer service voice assistant. Keep answers brief and conversational (under 3 sentences)."
    )

    # 2. Step 1: Provide User Audio (For demo, we generate user speech to audio/input/user_question.wav)
    user_input_audio = Path("audio/input/user_question.wav")
    simulated_user_text = "Hello! Can you tell me what services this voice automation chatbot offers?"
    
    print(f"\n[2/4] Step 1: Preparing User Audio Input...")
    print(f"      Simulating user speaking: \"{simulated_user_text}\"")
    print(f"      Generating audio file at : {user_input_audio}")
    
    start_total = time.perf_counter()
    await tts_service.text_to_speech(simulated_user_text, output_path=user_input_audio)
    print("      [OK] User speech WAV file ready.")

    # 3. Step 2: Speech-to-Text (Whisper API)
    print(f"\n[3/4] Step 2: Transcribing Audio with OpenAI Whisper...")
    stt_start = time.perf_counter()
    transcribed_text = await stt_service.speech_to_text(user_input_audio)
    stt_latency = time.perf_counter() - stt_start
    print(f"      Recognized Text : \"{transcribed_text}\"")
    print(f"      STT Latency     : {stt_latency:.2f}s")

    # 4. Step 3: LLM Reasoning (GPT-4o)
    print(f"\n[4/4] Step 3 & 4: LLM Reasoning -> Synthesizing Assistant Voice...")
    await conv_service.add_message(session_id, RoleEnum.USER, transcribed_text)
    history = await conv_service.get_history(session_id)

    llm_start = time.perf_counter()
    assistant_text = await llm_service.generate_response(
        user_message=transcribed_text,
        conversation_history=history,
        system_prompt="You are a friendly AI customer service voice assistant. Keep answers brief and conversational (under 3 sentences)."
    )
    llm_latency = time.perf_counter() - llm_start
    await conv_service.add_message(session_id, RoleEnum.ASSISTANT, assistant_text)

    # Step 4: Text-to-Speech (OpenAI TTS)
    assistant_output_audio = Path("audio/output/assistant_response.wav")
    tts_start = time.perf_counter()
    await tts_service.text_to_speech(assistant_text, output_path=assistant_output_audio)
    tts_latency = time.perf_counter() - tts_start
    total_latency = time.perf_counter() - start_total

    print("\n" + "=" * 70)
    print("  VOICE TURN EXECUTION COMPLETE!")
    print("=" * 70)
    print(f"User Spoke Text      : \"{transcribed_text}\"")
    print(f"Assistant Answer     : \"{assistant_text}\"")
    print("=" * 70)
    print(f"Input Audio File     : {user_input_audio.absolute()}")
    print(f"Output Audio File    : {assistant_output_audio.absolute()}")
    print("=" * 70)
    print("PERFORMANCE BREAKDOWN:")
    print(f"   * Whisper STT Latency : {stt_latency:.2f}s")
    print(f"   * GPT-4o LLM Latency  : {llm_latency:.2f}s")
    print(f"   * OpenAI TTS Latency  : {tts_latency:.2f}s")
    print(f"   * Total Turn Latency  : {total_latency:.2f}s")
    print("=" * 70)
    print("\nTip: Open the Output Audio File in any media player to listen to the chatbot speak!")


if __name__ == "__main__":
    asyncio.run(main())
