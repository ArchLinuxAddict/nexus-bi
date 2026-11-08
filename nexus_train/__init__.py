"""Standalone NEXUS-BI training runtime.

This package is intentionally independent of the interactive agent stack. It
reuses the unified model, objective, mission-policy and checkpoint code, but
adds a frozen-dataset gate, hardware safety, recovery and a terminal CLI.
"""

__version__ = "0.1.0"
