"""
Test Scenario Data Models.

Defines Pydantic validation schemas for conversational turns and full test scenarios
parsed from JSON or YAML files in `test_data/conversations/`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ScenarioTurn(BaseModel):
    """Single turn definition within an automated conversation test scenario."""
    model_config = ConfigDict(frozen=True)

    turn_id: int = Field(default=1, description="Sequential turn index (1-indexed).")
    user_prompt: str = Field(..., description="Text representation of user input prompt.")
    input_audio_path: Optional[Path | str] = Field(default=None, description="Optional path to WAV audio file for this turn.")
    expected_text: Optional[str] = Field(default=None, description="Expected ground-truth speech recognition text.")
    expected_answer: Optional[str] = Field(default=None, description="Expected semantic answer from the chatbot.")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt override for this turn.")


class ConversationScenario(BaseModel):
    """Complete specification for a multi-turn functional or regression test scenario."""
    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(..., description="Unique scenario identifier (e.g., SCENARIO_001).")
    scenario_name: str = Field(..., description="Human-readable title of the test scenario.")
    description: str = Field(..., description="Detailed explanation of the scenario objective.")
    turns: List[ScenarioTurn] = Field(default_factory=list, description="Chronological sequence of conversational turns to execute.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tags, categorization, or baseline info.")
