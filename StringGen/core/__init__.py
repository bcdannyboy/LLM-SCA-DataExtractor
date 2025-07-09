#!/usr/bin/env python3
"""
Core module for the Special Characters Attack (SCA) string generator.

This module provides the foundation for high-performance string generation
used in training data extraction attacks against Large Language Models.
"""

from .generator import StringGenerator
from .strategies import (
    INSET1Strategy,
    INSET2Strategy, 
    CROSS1Strategy,
    CROSS2Strategy,
    CROSS3Strategy,
    Strategy
)
from .character_sets import CharacterSets

__all__ = [
    'StringGenerator',
    'CharacterSets',
    'Strategy',
    'INSET1Strategy',
    'INSET2Strategy',
    'CROSS1Strategy', 
    'CROSS2Strategy',
    'CROSS3Strategy'
]

__version__ = '2.0.0'