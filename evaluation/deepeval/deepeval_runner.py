"""
Backward-compatibility re-export shim for DeepEvalRunner.
Please import from `evaluation.engines.deepeval.runner` in new code.
"""

from evaluation.engines.deepeval.runner import DeepEvalRunner

__all__ = ["DeepEvalRunner"]
