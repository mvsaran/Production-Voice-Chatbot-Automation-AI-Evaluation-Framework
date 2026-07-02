"""
AI Evaluation Framework Configuration.

Provides environment-driven configuration using Pydantic Settings for:
- DeepEval judge model parameters and API authentication
- Quality assurance thresholds (Correctness, Relevancy, Faithfulness, Hallucination, Toxicity)
- Speech recognition quality limits (Max Word Error Rate, Max Character Error Rate)
- Pipeline latency SLAs (STT, LLM, TTS, End-to-End)
- CI/CD execution flags and report storage locations
"""

from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


from app.config.settings import settings


class EvaluationSettings(BaseSettings):
    """
    Production-grade AI Evaluation Settings validated via Pydantic.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # DeepEval & Judge LLM Configuration
    EVAL_MODEL_NAME: str = Field(
        default="gpt-4o",
        description="LLM model used by DeepEval as the evaluator judge.",
    )
    OPENAI_API_KEY: SecretStr = Field(
        default=SecretStr("sk-mock-key"),
        description="OpenAI API key for evaluation metrics and LLM judge.",
    )
    DEEPEVAL_TELEMETRY_OPT_OUT: bool = Field(
        default=True,
        description="Disable DeepEval telemetry collection during CI/CD test runs.",
    )

    # AI Response Evaluation Thresholds (0.0 to 1.0)
    EVAL_THRESHOLD_CORRECTNESS: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score for Answer Correctness.",
    )
    EVAL_THRESHOLD_RELEVANCY: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score for Answer Relevancy.",
    )
    EVAL_THRESHOLD_FAITHFULNESS: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score for Faithfulness to context.",
    )
    EVAL_THRESHOLD_HALLUCINATION: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum tolerated score for Hallucination (lower is better).",
    )
    EVAL_THRESHOLD_TOXICITY: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum tolerated score for Toxicity and unsafe content.",
    )

    # Voice Quality Evaluation Limits
    EVAL_MAX_WER: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable Word Error Rate (WER) for speech recognition.",
    )
    EVAL_MAX_CER: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable Character Error Rate (CER) for speech recognition.",
    )

    # Latency Service Level Agreements (SLAs) in seconds
    EVAL_MAX_STT_LATENCY_SEC: float = Field(
        default=1.5,
        gt=0.0,
        description="Maximum allowed latency for Speech-To-Text transcription.",
    )
    EVAL_MAX_LLM_LATENCY_SEC: float = Field(
        default=3.0,
        gt=0.0,
        description="Maximum allowed latency for LLM response generation.",
    )
    EVAL_MAX_TTS_LATENCY_SEC: float = Field(
        default=2.0,
        gt=0.0,
        description="Maximum allowed latency for Text-To-Speech audio synthesis.",
    )
    EVAL_MAX_TOTAL_LATENCY_SEC: float = Field(
        default=6.0,
        gt=0.0,
        description="Maximum allowed end-to-end voice roundtrip latency.",
    )

    # Directory Paths for Test Data and Reports
    TEST_DATA_DIR: Path = Field(
        default=Path("test_data"),
        description="Root directory containing test prompts, expected answers, and conversation scenarios.",
    )
    REPORTS_HTML_DIR: Path = Field(
        default=Path("reports/html"),
        description="Directory where generated HTML quality assurance reports are saved.",
    )
    REPORTS_JSON_DIR: Path = Field(
        default=Path("reports/json"),
        description="Directory where structured JSON dashboard reports are saved.",
    )

    # CI/CD Pipeline Execution Flags
    FAIL_ON_REGRESSION: bool = Field(
        default=True,
        description="Whether test runner should exit with non-zero status if regression or metric failure is detected.",
    )

    @field_validator("TEST_DATA_DIR", "REPORTS_HTML_DIR", "REPORTS_JSON_DIR", mode="after")
    @classmethod
    def ensure_eval_directories_exist(cls, v: Path) -> Path:
        """Ensure all required evaluation and reporting directories exist on disk."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def get_openai_api_key_str(self) -> str:
        """Safely retrieve the OpenAI API key string from core settings or fallback."""
        try:
            return settings.OPENAI_API_KEY.get_secret_value()
        except Exception:
            return self.OPENAI_API_KEY.get_secret_value()


# Singleton instance exported for dependency injection across evaluation modules
eval_settings = EvaluationSettings()
