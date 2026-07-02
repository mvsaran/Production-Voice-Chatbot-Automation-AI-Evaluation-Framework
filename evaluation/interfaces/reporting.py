"""
Abstract QA Reporting Interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from evaluation.models import DashboardReadyEvaluationOutput


class BaseReportGenerator(ABC):
    """Abstract interface for compiling and exporting enterprise QA evaluation reports."""

    @abstractmethod
    def generate_html_report(
        self,
        results: List[DashboardReadyEvaluationOutput],
        output_file_name: str = "evaluation_report.html",
    ) -> Path:
        """Generate a styled, executive-friendly HTML QA report using Jinja2 templates."""
        pass

    @abstractmethod
    def generate_json_report(
        self,
        results: List[DashboardReadyEvaluationOutput],
        output_file_name: str = "evaluation_report.json",
    ) -> Path:
        """Export structured dashboard JSON data for CI/CD or visualization ingestion."""
        pass
