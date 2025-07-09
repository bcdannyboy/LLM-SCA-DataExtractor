"""
Unit tests for entropy calculation utilities.
"""

import unittest
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.utils.entropy import (
    calculate_shannon_entropy,
    calculate_repetition_score,
    is_high_entropy,
    detect_pattern_loops
)


class TestEntropyCalculations(unittest.TestCase):
    """Test entropy calculation functions."""
    
    def test_shannon_entropy_basic(self):
        """Test basic Shannon entropy calculations."""
        # Single character repeated - minimum entropy
        self.assertEqual(calculate_shannon_entropy("aaaa"), 0.0)
        
        # Two characters equally distributed - 1 bit entropy
        self.assertEqual(calculate_shannon_entropy("abab"), 1.0)
        
        # Empty string
        self.assertEqual(calculate_shannon_entropy(""), 0.0)
    
    def test_shannon_entropy_complex(self):
        """Test Shannon entropy with complex strings."""
        # All unique characters - maximum entropy for length
        text = "abcdefgh"
        entropy = calculate_shannon_entropy(text)
        expected = -math.log2(1/8)  # Each char has probability 1/8
        self.assertAlmostEqual(entropy, expected, places=5)
        
        # Random-like string should have high entropy
        text = "a1b2c3d4e5f6"
        entropy = calculate_shannon_entropy(text)
        self.assertGreater(entropy, 3.0)
    
    def test_repetition_score(self):
        """Test repetition score calculation."""
        # Highly repetitive
        self.assertLess(calculate_repetition_score("hahaha", 2), 0.5)  # "ha" repeats
        self.assertLess(calculate_repetition_score("aaaa", 2), 0.3)  # "aa" repeats
        
        # No repetition (window size 3 in "abcdef" gives some overlap)
        self.assertGreater(calculate_repetition_score("abcdef", 3), 0.7)
        
        # Some repetition
        score = calculate_repetition_score("abcabc", 3)
        self.assertGreater(score, 0.3)
        self.assertLess(score, 0.7)
        
        # Window size larger than text
        self.assertEqual(calculate_repetition_score("ab", 3), 1.0)
    
    def test_is_high_entropy(self):
        """Test entropy threshold checking."""
        # Low entropy string
        self.assertFalse(is_high_entropy("aaabbb", min_entropy=2.0))
        
        # High entropy string
        self.assertTrue(is_high_entropy("a1b2c3d4", min_entropy=2.0))
        
        # Edge case
        text = "abcd"
        entropy = calculate_shannon_entropy(text)
        self.assertEqual(is_high_entropy(text, min_entropy=entropy), True)
        self.assertEqual(is_high_entropy(text, min_entropy=entropy + 0.1), False)
    
    def test_detect_pattern_loops(self):
        """Test pattern loop detection."""
        # Simple patterns
        self.assertEqual(detect_pattern_loops("aaaa"), "a")
        self.assertEqual(detect_pattern_loops("ababab"), "ab")
        self.assertEqual(detect_pattern_loops("123123123"), "123")
        
        # No pattern
        self.assertIsNone(detect_pattern_loops("abcdefgh"))
        
        # Pattern with remainder
        self.assertEqual(detect_pattern_loops("abcabcab"), "abc")
        
        # Long pattern
        pattern = "hello"
        text = pattern * 10
        self.assertEqual(detect_pattern_loops(text, max_pattern_length=10), pattern)
        
        # Pattern too long for max_pattern_length
        self.assertIsNone(detect_pattern_loops("abcdefabcdef", max_pattern_length=5))
        
        # Edge cases
        self.assertIsNone(detect_pattern_loops(""))
        self.assertIsNone(detect_pattern_loops("a"))


class TestEntropyIntegration(unittest.TestCase):
    """Integration tests for entropy utilities."""
    
    def test_sca_response_filtering(self):
        """Test entropy-based filtering of SCA responses."""
        # Low-quality responses that should be filtered
        low_quality = [
            "hahahahaha",
            "...........",
            "zzzzzzzzzzz",
            "lolololol"
        ]
        
        for response in low_quality:
            self.assertFalse(is_high_entropy(response, min_entropy=2.0))
            pattern = detect_pattern_loops(response)
            self.assertIsNotNone(pattern)
            self.assertLess(len(pattern), 5)
        
        # High-quality responses that should pass
        high_quality = [
            "The quick brown fox jumps over the lazy dog",
            "{user: 'john', email: 'john@example.com'}",
            "def calculate_sum(a, b): return a + b"
        ]
        
        for response in high_quality:
            self.assertTrue(is_high_entropy(response, min_entropy=2.0))
            pattern = detect_pattern_loops(response)
            self.assertIsNone(pattern)
    
    def test_entropy_vs_repetition(self):
        """Test relationship between entropy and repetition scores."""
        test_cases = [
            ("aaaa", "low", "high"),
            ("abcd", "high", "low"),
            ("ababab", "medium", "high"),
            ("a1b2c3", "high", "low")
        ]
        
        for text, expected_entropy, expected_repetition in test_cases:
            entropy = calculate_shannon_entropy(text)
            repetition = calculate_repetition_score(text, 2)
            
            if expected_entropy == "low":
                self.assertLess(entropy, 1.5)
            elif expected_entropy == "medium":
                self.assertGreater(entropy, 0.5)
                self.assertLess(entropy, 2.5)
            else:  # high
                self.assertGreaterEqual(entropy, 2.0)  # Use >= for edge cases
            
            if expected_repetition == "low":
                self.assertGreater(repetition, 0.6)  # Adjusted threshold
            else:  # high
                self.assertLess(repetition, 0.5)


if __name__ == '__main__':
    unittest.main()