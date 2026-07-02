"""
Abstract Test Runner Interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from evaluation.models import ConversationScenario, TurnExecutionArtifacts


class BaseTestRunner(ABC):
    """Abstract interface for automated test execution engines (Functional, Regression, Negative)."""

    @abstractmethod
    async def run_scenario(
        self,
        scenario: ConversationScenario,
    ) -> List[TurnExecutionArtifacts]:
        """Automatically drive an end-to-end multi-turn conversation scenario."""
        pass

    @abstractmethod
    def load_scenarios_from_file(self, file_path: Path | str) -> List[ConversationScenario]:
        """Parse test scenario definitions from JSON, YAML, or CSV files."""
        pass
