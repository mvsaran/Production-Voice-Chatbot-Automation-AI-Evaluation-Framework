"""
Live Microphone Voice Assistant Script (Talk Directly to Your Bot!).

This script allows you to speak into your computer microphone, transcribes your
voice command using Whisper, gets a response from GPT-4o, synthesizes the voice
using OpenAI TTS, and automatically plays the response back through your speakers!
"""

import asyncio
from pathlib import Path
import sys
import time
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
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

# Audio recording parameters
SAMPLE_RATE = 16000  # 16kHz is ideal for Whisper STT
RECORD_DURATION = 5  # Record 5 seconds of speech per turn


def record_microphone_audio(output_path: Path, duration: int = RECORD_DURATION, fs: int = SAMPLE_RATE) -> None:
    """Record mono audio from the primary computer microphone and save as WAV."""
    print(f"\n🎙️  LISTENING... Speak your command clearly now! (Recording for {duration} seconds)...")
    
    # Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
    
    # Show a visual countdown spinner
    for i in range(duration, 0, -1):
        print(f"    ⏳ {i}s remaining...", end="\r", flush=True)
        time.sleep(1)
    
    sd.wait()  # Wait until recording is finished
    print("    ✅ Recording captured! Saving to disk...               ")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(output_path), fs, recording)
    print(f"    📁 Saved voice command to: {output_path}")


def play_audio_response(audio_path: Path) -> None:
    """Play a WAV audio file through the computer's primary speakers."""
    print(f"\n🔊 PLAYING BOT VOICE RESPONSE through your speakers...")
    try:
        fs, data = wav.read(str(audio_path))
        sd.play(data, fs)
        sd.wait()
        print("    ✅ Audio playback finished!")
    except Exception as e:
        print(f"    ⚠️ Could not play audio automatically ({e}). You can open {audio_path} manually!")


async def main():
    print("=" * 70)
    print("  TALK TO YOUR BOT - LIVE MICROPHONE INTERACTION ENGINE")
    print("=" * 70)

    # 1. Initialize Voice Services
    print("\n[1/4] Initializing Voice & AI Services...")
    conv_service = InMemoryConversationService()
    stt_service = OpenAIWhisperService()
    llm_service = OpenAILLMService()
    tts_service = OpenAITTSService()
    print("      [OK] Whisper STT, GPT-4o LLM, and OpenAI TTS loaded.")

    session_id = "LIVE_MIC_SESSION_001"
    await conv_service.create_conversation(
        conversation_id=session_id,
        system_prompt="You are a helpful and articulate AI voice assistant. Speak concisely and conversationally."
    )

    while True:
        print("\n" + "-" * 70)
        user_choice = input("👉 Press [ENTER] to record your voice command (or type 'q' and ENTER to quit): ").strip().lower()
        if user_choice == "q":
            print("\n👋 Exiting Live Microphone Assistant. Goodbye!")
            break

        # 2. Record Microphone Input
        input_audio_path = Path("audio/input/live_mic_command.wav")
        try:
            record_microphone_audio(input_audio_path, duration=RECORD_DURATION)
        except Exception as e:
            print(f"\n❌ Microphone Error: Could not record audio ({e}). Check your microphone permissions!")
            continue

        # 3. Transcribe with Whisper
        print("\n[2/4] Transcribing your voice command with Whisper...")
        try:
            user_text = await stt_service.speech_to_text(input_audio_path)
            if not user_text or user_text.strip() == "":
                print("⚠️ No speech recognized! Please speak louder or closer to the mic.")
                continue
            print(f"      👤 YOU SAID: \"{user_text}\"")
        except Exception as e:
            print(f"❌ Speech-to-Text failed: {e}")
            continue

        # 4. Generate LLM Answer
        print("\n[3/4] Generating intelligent reasoning from GPT-4o...")
        await conv_service.add_message(session_id, RoleEnum.USER, user_text)
        history = await conv_service.get_history(session_id)
        
        try:
            bot_answer = await llm_service.generate_response(
                user_message=user_text,
                conversation_history=history,
                system_prompt="You are a helpful and articulate AI voice assistant. Speak concisely and conversationally."
            )
            await conv_service.add_message(session_id, RoleEnum.ASSISTANT, bot_answer)
            print(f"      🤖 BOT ANSWER: \"{bot_answer}\"")
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            continue

        # 5. Synthesize Bot Voice & Play It
        print("\n[4/4] Synthesizing bot voice with OpenAI TTS...")
        output_audio_path = Path("audio/output/live_bot_response.wav")
        try:
            await tts_service.text_to_speech(bot_answer, output_path=output_audio_path)
            print(f"      📁 Saved bot speech to: {output_audio_path}")
            play_audio_response(output_audio_path)
        except Exception as e:
            print(f"❌ Text-to-Speech failed: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
