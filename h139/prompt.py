"""H139 shared frozen native START/END point-grounding instructions.

Use the exact established H136 prompt contract in both baseline and
trained arms; never leak DOM geometry, labels, or target coordinates.
"""
from h136.prompt import role_prompt

__all__ = ["role_prompt"]
