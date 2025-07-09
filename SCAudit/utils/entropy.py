"""
Entropy calculation utilities for SCA analysis.

This module provides functions to calculate Shannon entropy and related
metrics for filtering low-quality responses.
"""

import math
from collections import Counter
from typing import Optional


def calculate_shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text in bits per character.
    
    Shannon entropy measures the average information content per character.
    Higher entropy indicates more randomness/diversity in the text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        float: Shannon entropy in bits per character
    """
    if not text:
        return 0.0
    
    # Count character frequencies
    char_counts = Counter(text)
    total_chars = len(text)
    
    # Calculate probabilities and entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_repetition_score(text: str, window_size: int = 3) -> float:
    """
    Calculate a repetition score based on repeated substrings.
    
    Lower scores indicate more repetition (e.g., "hahaha" patterns).
    
    Args:
        text: Input text to analyze
        window_size: Size of substring window to check
        
    Returns:
        float: Repetition score (0.0 = highly repetitive, 1.0 = no repetition)
    """
    if len(text) < window_size:
        return 1.0
    
    # Count occurrences of each substring
    substrings = Counter()
    for i in range(len(text) - window_size + 1):
        substring = text[i:i + window_size]
        substrings[substring] += 1
    
    # Calculate repetition based on most common substring
    if not substrings:
        return 1.0
    
    max_count = max(substrings.values())
    possible_positions = len(text) - window_size + 1
    
    # Normalize to 0-1 range (inverted so high repetition = low score)
    repetition_ratio = max_count / possible_positions
    return 1.0 - repetition_ratio


def is_high_entropy(text: str, min_entropy: float = 2.0) -> bool:
    """
    Check if text has sufficiently high entropy.
    
    Args:
        text: Input text to check
        min_entropy: Minimum entropy threshold in bits/char
        
    Returns:
        bool: True if entropy is above threshold
    """
    return calculate_shannon_entropy(text) >= min_entropy


def detect_pattern_loops(text: str, max_pattern_length: int = 20) -> Optional[str]:
    """
    Detect obvious pattern loops in text.
    
    Args:
        text: Input text to analyze
        max_pattern_length: Maximum pattern length to check
        
    Returns:
        Optional[str]: The detected pattern if found, None otherwise
    """
    if len(text) < 2:
        return None
    
    # Check for patterns of increasing length
    for pattern_len in range(1, min(max_pattern_length, len(text) // 2) + 1):
        pattern = text[:pattern_len]
        
        # Check if the entire text is just repetitions of this pattern
        repetitions = len(text) // pattern_len
        remainder = len(text) % pattern_len
        
        # Check if pattern repeats through the text
        expected = pattern * repetitions + pattern[:remainder]
        if expected == text:
            return pattern
        
        # Also check with 80% coverage threshold for partial matches
        if pattern * repetitions == text[:pattern_len * repetitions]:
            if pattern_len * repetitions >= len(text) * 0.8:
                return pattern
    
    return None