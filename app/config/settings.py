"""
Application Configuration settings using Pydantic Settings.

Supports environment variables, .env file loading, and validation for OpenAI,
Azure OpenAI, Speech models, audio directory paths, and logging configurations.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Production-ready application settings validated via Pydantic.
    
    Provides easy configuration switching between standard OpenAI and Azure OpenAI,
    as well as configurable speech models and file storage locations.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # OpenAI Authentication
    OPENAI_API_KEY: SecretStr = Field(
        default=SecretStr("sk-mock-key"),
        description="Standard OpenAI API key.",
    )

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENABLED: bool = Field(
        default=False,
        description="Flag to toggle Azure OpenAI integration instead of standard OpenAI.",
    )
    AZURE_OPENAI_API_KEY: Optional[SecretStr] = Field(
        default=None,
        description="Azure OpenAI API key when Azure is enabled.",
    )
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL.",
    )
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version.",
    )
    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name for LLM completion.",
    )

    # Model Parameters
    MODEL_NAME: str = Field(
        default="gpt-4o",
        description="LLM model name (e.g., gpt-4o, gpt-4.1).",
    )
    WHISPER_MODEL: str = Field(
        default="whisper-1",
        description="Speech-To-Text Whisper API model name.",
    )
    TTS_MODEL: str = Field(
        default="tts-1",
        description="Text-To-Speech OpenAI API model name.",
    )
    TTS_VOICE: str = Field(
        default="alloy",
        description="Default voice for text-to-speech generation (e.g., alloy, echo, fable, onyx, nova, shimmer).",
    )

    # Prompt Engineering
    DEFAULT_SYSTEM_PROMPT: str = Field(
        default="You are a helpful, friendly voice assistant. Keep your answers concise, conversational, and natural for spoken interaction.",
        description="Default system prompt for conversation context.",
    )

    # Audio Storage Locations
    AUDIO_INPUT_DIR: Path = Field(
        default=Path("audio/input"),
        description="Directory path where recorded input WAV audio files are stored.",
    )
    AUDIO_OUTPUT_DIR: Path = Field(
        default=Path("audio/output"),
        description="Directory path where generated output audio files are stored.",
    )

    # Logging Parameters
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    LOG_FILE_PATH: Path = Field(
        default=Path("app/logs/voice_automation.log"),
        description="File path for persistent log storage and rotation.",
    )

    @field_validator("AUDIO_INPUT_DIR", "AUDIO_OUTPUT_DIR", mode="after")
    @classmethod
    def ensure_directories_exist(cls, v: Path) -> Path:
        """Ensure that configured audio directories exist on the filesystem."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("LOG_FILE_PATH", mode="after")
    @classmethod
    def ensure_log_directory_exists(cls, v: Path) -> Path:
        """Ensure that the parent directory for log files exists."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    def get_openai_api_key_str(self) -> str:
        """Safely retrieve the OpenAI API key as a plaintext string."""
        return self.OPENAI_API_KEY.get_secret_value()

    def get_azure_api_key_str(self) -> Optional[str]:
        """Safely retrieve the Azure OpenAI API key as a plaintext string if configured."""
        if self.AZURE_OPENAI_API_KEY:
            return self.AZURE_OPENAI_API_KEY.get_secret_value()
        return None


# Singleton settings instance to be injected across modules
settings = Settings()
