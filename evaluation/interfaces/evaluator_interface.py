"""
Backward-compatibility re-export shim for Evaluator interface.
Please import from `evaluation.interfaces.evaluator` in new code.
"""

from evaluation.interfaces.evaluator import Evaluator

__all__ = ["Evaluator"]
