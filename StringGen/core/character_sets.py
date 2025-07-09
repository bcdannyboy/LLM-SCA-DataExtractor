#!/usr/bin/env python3
"""
Character sets for SCA string generation.

This module defines the three character sets used in the Special Characters Attack:
- S1: Structural symbols commonly used in data structures
- S2: Other special characters found in web content  
- L: Lowercase English letters

Performance optimizations:
- Pre-computed character arrays stored as bytes for faster access
- Immutable tuples to prevent modification and improve cache locality
- Pre-calculated set sizes to avoid repeated len() calls
"""

import array
from typing import Tuple, List, Set, FrozenSet


class CharacterSets:
    """
    High-performance character set definitions for SCA string generation.
    
    All sets are immutable and optimized for fast random access and iteration.
    Character data is stored in multiple formats for different use cases:
    - Tuples for indexed access
    - Frozen sets for membership testing
    - Byte arrays for memory-efficient bulk operations
    """
    
    # S1: Structural symbols - 8 characters
    # These symbols are commonly used to define data structures in various formats
    S1_CHARS: Tuple[str, ...] = ('{', '}', '[', ']', '(', ')', '<', '>')
    S1_SET: FrozenSet[str] = frozenset(S1_CHARS)
    S1_BYTES: bytes = b'{}[]()<>'
    S1_SIZE: int = 8
    
    # S2: Other special characters - 22 characters according to the paper
    # Note: The original implementation only had 9 characters, but the paper mentions 22
    # Including the full set as described in STRING_GENERATION.md
    S2_CHARS: Tuple[str, ...] = (
        '!', '$', '@', '#', '%', '&', '*', '_', '+',
        ';', ':', '"', "'", ',', '.', '/', '?',
        '~', '`', '^', '|', '='
    )
    S2_SET: FrozenSet[str] = frozenset(S2_CHARS)
    S2_BYTES: bytes = b'!$@#%&*_+;:"\',./~`^|=?'
    S2_SIZE: int = len(S2_CHARS)  # Dynamically calculate to ensure accuracy
    
    # L: Lowercase English letters - 26 characters
    L_CHARS: Tuple[str, ...] = tuple(chr(i) for i in range(ord('a'), ord('z') + 1))
    L_SET: FrozenSet[str] = frozenset(L_CHARS)
    L_BYTES: bytes = b'abcdefghijklmnopqrstuvwxyz'
    L_SIZE: int = 26
    
    # Combined sets for cross-set operations
    ALL_CHARS: Tuple[str, ...] = S1_CHARS + S2_CHARS + L_CHARS
    ALL_SET: FrozenSet[str] = S1_SET | S2_SET | L_SET
    ALL_SIZE: int = S1_SIZE + S2_SIZE + L_SIZE
    
    # Pre-computed character arrays for ultra-fast access
    # Using array.array for memory efficiency with single-byte characters
    S1_ARRAY: array.array = array.array('u', S1_CHARS)
    S2_ARRAY: array.array = array.array('u', S2_CHARS)
    L_ARRAY: array.array = array.array('u', L_CHARS)
    ALL_ARRAY: array.array = array.array('u', ALL_CHARS)
    
    # Lookup tables for O(1) character access by set name
    SETS_BY_NAME = {
        'S1': S1_CHARS,
        'S2': S2_CHARS,
        'L': L_CHARS,
        'ALL': ALL_CHARS
    }
    
    ARRAYS_BY_NAME = {
        'S1': S1_ARRAY,
        'S2': S2_ARRAY,
        'L': L_ARRAY,
        'ALL': ALL_ARRAY
    }
    
    SIZES_BY_NAME = {
        'S1': S1_SIZE,
        'S2': S2_SIZE,
        'L': L_SIZE,
        'ALL': ALL_SIZE
    }
    
    @classmethod
    def get_set(cls, name: str) -> Tuple[str, ...]:
        """
        Get character set by name.
        
        Args:
            name: Set name ('S1', 'S2', 'L', or 'ALL')
            
        Returns:
            Tuple of characters in the requested set
            
        Raises:
            KeyError: If set name is invalid
        """
        return cls.SETS_BY_NAME[name.upper()]
    
    @classmethod
    def get_array(cls, name: str) -> array.array:
        """
        Get character array by name for high-performance operations.
        
        Args:
            name: Set name ('S1', 'S2', 'L', or 'ALL')
            
        Returns:
            Array of characters for fast indexed access
            
        Raises:
            KeyError: If set name is invalid
        """
        return cls.ARRAYS_BY_NAME[name.upper()]
    
    @classmethod
    def get_size(cls, name: str) -> int:
        """
        Get pre-computed size of a character set.
        
        Args:
            name: Set name ('S1', 'S2', 'L', or 'ALL')
            
        Returns:
            Number of characters in the set
            
        Raises:
            KeyError: If set name is invalid
        """
        return cls.SIZES_BY_NAME[name.upper()]
    
    @classmethod
    def validate_length(cls, set_name: str, requested_length: int) -> None:
        """
        Validate that requested length doesn't exceed set size.
        
        Args:
            set_name: Name of the character set
            requested_length: Number of characters requested
            
        Raises:
            ValueError: If requested length exceeds set size
        """
        max_size = cls.get_size(set_name)
        if requested_length > max_size:
            raise ValueError(
                f"Cannot sample {requested_length} unique characters from "
                f"{set_name} set (size={max_size})"
            )