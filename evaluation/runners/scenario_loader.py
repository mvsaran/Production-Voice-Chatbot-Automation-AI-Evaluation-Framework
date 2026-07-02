"""
Scenario Loader Utility.

Responsible for reading and validating test scenario definitions from JSON or YAML files.
Extracted into a standalone module to adhere to Single Responsibility Principle (SRP).
"""

import json
from pathlib import Path
from typing import Any, Dict
import yaml
from loguru import logger


class ScenarioLoader:
    """Utility class for loading and validating conversation scenario files."""

    @classmethod
    def load_scenario(cls, file_path: Path | str) -> Dict[str, Any]:
        """
        Load a test scenario definition from JSON or YAML file.

        Args:
            file_path: Path to `.json` or `.yaml`/`.yml` scenario file.

        Returns:
            Dictionary containing scenario definition.

        Raises:
            FileNotFoundError: If scenario file does not exist.
            ValueError: If JSON/YAML parsing fails or syntax is invalid.
        """
        path = Path(file_path)
        logger.info(f"Loading test scenario from: {path}")

        if not path.exists():
            err = f"Missing scenario file: {path}"
            logger.error(err)
            raise FileNotFoundError(err)

        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                data = json.loads(content)
            elif path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(content)
            else:
                err = f"Unsupported scenario file extension '{path.suffix}'. Use .json or .yaml."
                logger.error(err)
                raise ValueError(err)

            required_keys = {"conversation_id", "description", "conversation"}
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                err = f"Scenario definition in '{path}' missing required fields: {missing_keys}"
                logger.error(err)
                raise ValueError(err)

            if not isinstance(data["conversation"], list):
                err = f"Scenario field 'conversation' must be a list in '{path}'."
                logger.error(err)
                raise ValueError(err)

            return data
        except json.JSONDecodeError as e:
            err = f"Invalid JSON in scenario file '{path}': {str(e)}"
            logger.error(err)
            raise ValueError(err) from e
        except yaml.YAMLError as e:
            err = f"Invalid YAML in scenario file '{path}': {str(e)}"
            logger.error(err)
            raise ValueError(err) from e
