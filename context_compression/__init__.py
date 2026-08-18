"""
context_compression — A three-layer context management engine for LLM agents.

Structured state extraction · Tool output offloading · History compression
"""

__version__ = "1.0.0"

from .config import CCMConfig  # noqa: F401
from .global_state import GenericState  # noqa: F401
from .agent import run_agent  # noqa: F401
