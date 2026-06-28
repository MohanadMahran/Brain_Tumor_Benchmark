"""Evaluation modules for benchmark testing.

Provides:
    - Sliding window inference evaluator
    - Report generation
"""

from evaluation.evaluator import BenchmarkEvaluator
from evaluation.report import generate_comparison_report

__all__ = ["BenchmarkEvaluator", "generate_comparison_report"]
