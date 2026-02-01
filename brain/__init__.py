"""
Brain v3 - AI Agent Memory System

Persistent, encrypted memory for AI agents with Soul, Bonding, and Semantic Search.

Install:
    pip install git+https://github.com/clawcolab/brain-v3.git

Usage:
    from brain import Brain, Memory, Soul, Bond, Goal
    
    brain = Brain()
"""

from .brain_v3 import Brain, Memory, Todo, Soul, Bond, Goal

__version__ = "3.0.0"
__all__ = ["Brain", "Memory", "Todo", "Soul", "Bond", "Goal"]
