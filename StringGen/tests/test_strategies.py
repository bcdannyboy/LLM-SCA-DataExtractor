#!/usr/bin/env python3
"""
Tests for SCA string generation strategies.

Validates that each strategy correctly implements the algorithms
described in the paper.
"""

import unittest
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.strategies import (
    INSET1Strategy, INSET2Strategy, CROSS1Strategy, 
    CROSS2Strategy, CROSS3Strategy, create_strategy
)
from core.character_sets import CharacterSets


class TestINSET1Strategy(unittest.TestCase):
    """Test INSET1: Repeat single character n times."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.strategy = INSET1Strategy(random.Random(42))
    
    def test_single_character_repetition(self):
        """Test that INSET1 produces strings of single repeated character."""
        for length in [1, 10, 100]:
            result = self.strategy.generate(length)
            self.assertEqual(len(result), length)
            # All characters should be the same
            self.assertEqual(len(set(result)), 1)
            # Character should be from one of the sets
            char = result[0]
            self.assertTrue(
                char in CharacterSets.S1_SET or
                char in CharacterSets.S2_SET or
                char in CharacterSets.L_SET
            )
    
    def test_all_variants_generation(self):
        """Test that generate_all_variants produces all possible combinations."""
        length = 5
        all_variants = list(self.strategy.generate_all_variants(length))
        
        # Should have one variant per character
        expected_count = (
            CharacterSets.S1_SIZE + 
            CharacterSets.S2_SIZE + 
            CharacterSets.L_SIZE
        )
        self.assertEqual(len(all_variants), expected_count)
        
        # Check each variant
        seen_chars = set()
        for set_name, sequence in all_variants:
            self.assertEqual(len(sequence), length)
            self.assertEqual(len(set(sequence)), 1)
            char = sequence[0]
            self.assertNotIn(char, seen_chars)  # No duplicates
            seen_chars.add(char)
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        strategy1 = INSET1Strategy(random.Random(123))
        strategy2 = INSET1Strategy(random.Random(123))
        
        for _ in range(10):
            self.assertEqual(
                strategy1.generate(50),
                strategy2.generate(50)
            )


class TestINSET2Strategy(unittest.TestCase):
    """Test INSET2: n unique random characters from one set."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.strategy = INSET2Strategy(random.Random(42))
    
    def test_single_set_characters(self):
        """Test that all characters come from same set."""
        for _ in range(10):
            result = self.strategy.generate(10)
            self.assertEqual(len(result), 10)
            
            # Check which set the characters belong to
            in_s1 = all(c in CharacterSets.S1_SET for c in result)
            in_s2 = all(c in CharacterSets.S2_SET for c in result)
            in_l = all(c in CharacterSets.L_SET for c in result)
            
            # Should be from exactly one set
            self.assertEqual(sum([in_s1, in_s2, in_l]), 1)
    
    def test_unique_characters_within_limit(self):
        """Test unique character selection when length <= set size."""
        # Use small length that fits in S1 (8 chars)
        result = self.strategy.generate(5)
        # If from S1, should have 5 unique characters
        if all(c in CharacterSets.S1_SET for c in result):
            self.assertEqual(len(set(result)), 5)
    
    def test_repetition_when_exceeding_set_size(self):
        """Test that characters repeat when length > set size."""
        # Generate more characters than in any single set
        result = self.strategy.generate(50)
        self.assertEqual(len(result), 50)
        
        # Characters should be from one set
        in_s1 = all(c in CharacterSets.S1_SET for c in result)
        in_s2 = all(c in CharacterSets.S2_SET for c in result)
        in_l = all(c in CharacterSets.L_SET for c in result)
        self.assertEqual(sum([in_s1, in_s2, in_l]), 1)


class TestCROSS1Strategy(unittest.TestCase):
    """Test CROSS1: n unique random characters across all sets."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.strategy = CROSS1Strategy(random.Random(42))
    
    def test_characters_from_all_sets(self):
        """Test that characters can come from any set."""
        # Generate many sequences to ensure we see all sets
        all_chars = set()
        for _ in range(100):
            result = self.strategy.generate(10)
            all_chars.update(result)
        
        # Should eventually see characters from all sets
        has_s1 = any(c in CharacterSets.S1_SET for c in all_chars)
        has_s2 = any(c in CharacterSets.S2_SET for c in all_chars)
        has_l = any(c in CharacterSets.L_SET for c in all_chars)
        
        self.assertTrue(has_s1)
        self.assertTrue(has_s2)
        self.assertTrue(has_l)
    
    def test_unique_characters_within_limit(self):
        """Test unique character selection when length <= total size."""
        # Total size is 8 + 22 + 26 = 56
        result = self.strategy.generate(30)
        self.assertEqual(len(result), 30)
        self.assertEqual(len(set(result)), 30)  # All unique
    
    def test_all_characters_valid(self):
        """Test that all generated characters are from defined sets."""
        for _ in range(10):
            result = self.strategy.generate(100)
            for char in result:
                self.assertTrue(
                    char in CharacterSets.ALL_SET,
                    f"Invalid character: {char}"
                )


class TestCROSS2Strategy(unittest.TestCase):
    """Test CROSS2: Distributed sampling with concatenation."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.strategy = CROSS2Strategy(random.Random(42))
    
    def test_structured_distribution(self):
        """Test that characters are distributed across sets as specified."""
        # Test with length divisible by 3
        result = self.strategy.generate(30)
        self.assertEqual(len(result), 30)
        
        # First 10 chars should be from S1
        part1 = result[:10]
        self.assertTrue(all(c in CharacterSets.S1_SET for c in part1))
        
        # Next 10 chars should be from S2
        part2 = result[10:20]
        self.assertTrue(all(c in CharacterSets.S2_SET for c in part2))
        
        # Last 10 chars should be from L
        part3 = result[20:30]
        self.assertTrue(all(c in CharacterSets.L_SET for c in part3))
    
    def test_remainder_distribution(self):
        """Test correct handling of remainder when length % 3 != 0."""
        # Length 31 = 10 + 10 + 11
        result = self.strategy.generate(31)
        self.assertEqual(len(result), 31)
        
        # Check part sizes (S1 gets extra char)
        part1 = result[:11]
        part2 = result[11:21]
        part3 = result[21:31]
        
        self.assertTrue(all(c in CharacterSets.S1_SET for c in part1))
        self.assertTrue(all(c in CharacterSets.S2_SET for c in part2))
        self.assertTrue(all(c in CharacterSets.L_SET for c in part3))
        
        # Length 32 = 11 + 11 + 10
        result = self.strategy.generate(32)
        self.assertEqual(len(result), 32)


class TestCROSS3Strategy(unittest.TestCase):
    """Test CROSS3: Shuffled CROSS2."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.strategy = CROSS3Strategy(random.Random(42))
    
    def test_character_distribution(self):
        """Test that CROSS3 has same character distribution as CROSS2."""
        length = 30
        result = self.strategy.generate(length)
        self.assertEqual(len(result), length)
        
        # Count characters from each set
        s1_count = sum(1 for c in result if c in CharacterSets.S1_SET)
        s2_count = sum(1 for c in result if c in CharacterSets.S2_SET)
        l_count = sum(1 for c in result if c in CharacterSets.L_SET)
        
        # Should have equal distribution (10 each for length 30)
        self.assertEqual(s1_count, 10)
        self.assertEqual(s2_count, 10)
        self.assertEqual(l_count, 10)
    
    def test_shuffling_effectiveness(self):
        """Test that characters are actually shuffled."""
        # Generate multiple sequences
        sequences = [self.strategy.generate(30) for _ in range(10)]
        
        # They should all be different due to shuffling
        unique_sequences = set(sequences)
        self.assertEqual(len(unique_sequences), 10)
        
        # Characters should not be in structured blocks
        for seq in sequences:
            # Check that not all S1 chars are at the beginning
            first_10 = seq[:10]
            s1_in_first_10 = sum(1 for c in first_10 if c in CharacterSets.S1_SET)
            # Unlikely to have all 10 S1 chars in first 10 positions
            self.assertLess(s1_in_first_10, 10)


class TestStrategyFactory(unittest.TestCase):
    """Test strategy factory function."""
    
    def test_create_all_strategies(self):
        """Test that all strategies can be created by name."""
        strategies = ['INSET1', 'INSET2', 'CROSS1', 'CROSS2', 'CROSS3']
        
        for name in strategies:
            strategy = create_strategy(name)
            self.assertIsNotNone(strategy)
            self.assertEqual(strategy.name, name)
            
            # Test that it can generate
            result = strategy.generate(10)
            self.assertEqual(len(result), 10)
    
    def test_case_insensitive(self):
        """Test that strategy names are case-insensitive."""
        strategy1 = create_strategy('inset1')
        strategy2 = create_strategy('INSET1')
        strategy3 = create_strategy('InSeT1')
        
        self.assertEqual(strategy1.name, 'INSET1')
        self.assertEqual(strategy2.name, 'INSET1')
        self.assertEqual(strategy3.name, 'INSET1')
    
    def test_invalid_strategy_name(self):
        """Test that invalid strategy names raise ValueError."""
        with self.assertRaises(ValueError):
            create_strategy('INVALID')
        
        with self.assertRaises(ValueError):
            create_strategy('')


if __name__ == '__main__':
    unittest.main()