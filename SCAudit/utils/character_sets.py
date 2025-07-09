"""
Character set definitions for SCA analysis.

This module defines the special character sets used in the SCA methodology
and provides utilities for analyzing character distributions in text.
"""

from typing import Set, Dict
import string


# Character set definitions from the SCA paper
S1: Set[str] = set("{}[]()<>")  # Structural symbols (8 chars)
S2: Set[str] = set("!$@#%&*_+-=|\\:;\"',.?/~`^")  # Special characters (24 chars - includes $ and @)
L: Set[str] = set(string.ascii_lowercase)  # Lowercase letters (26 chars)

# Combined sets
VS1: Set[str] = S1  # Visual Set 1
VS2: Set[str] = S2  # Visual Set 2
ALL_SPECIAL: Set[str] = S1 | S2  # All special characters
ALL_CHARS: Set[str] = S1 | S2 | L  # All SCA characters


def calculate_special_char_ratio(text: str) -> float:
    """
    Calculate the ratio of special characters in text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        float: Ratio of special characters (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    special_count = sum(1 for char in text if char in ALL_SPECIAL)
    return special_count / len(text)


def calculate_char_distribution(text: str) -> Dict[str, float]:
    """
    Calculate character distribution across SCA character sets.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dict mapping set names to their proportions in the text
    """
    if not text:
        return {"S1": 0.0, "S2": 0.0, "L": 0.0, "other": 0.0}
    
    counts = {"S1": 0, "S2": 0, "L": 0, "other": 0}
    
    for char in text:
        if char in S1:
            counts["S1"] += 1
        elif char in S2:
            counts["S2"] += 1
        elif char in L:
            counts["L"] += 1
        else:
            counts["other"] += 1
    
    total = len(text)
    return {k: v / total for k, v in counts.items()}


def contains_sca_pattern(text: str, min_special_ratio: float = 0.15) -> bool:
    """
    Check if text contains SCA-like patterns.
    
    Args:
        text: Input text to check
        min_special_ratio: Minimum ratio of special characters
        
    Returns:
        bool: True if text matches SCA patterns
    """
    return calculate_special_char_ratio(text) >= min_special_ratio