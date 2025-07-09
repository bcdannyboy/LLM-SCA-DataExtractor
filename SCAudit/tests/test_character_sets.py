"""
Unit tests for character set utilities.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.utils.character_sets import (
    S1, S2, L, ALL_SPECIAL, ALL_CHARS,
    calculate_special_char_ratio,
    calculate_char_distribution,
    contains_sca_pattern
)


class TestCharacterSets(unittest.TestCase):
    """Test character set definitions and utilities."""
    
    def test_character_set_sizes(self):
        """Test that character sets have correct sizes."""
        self.assertEqual(len(S1), 8)  # Structural symbols
        self.assertEqual(len(S2), 24)  # Special characters
        self.assertEqual(len(L), 26)  # Lowercase letters
        self.assertEqual(len(ALL_SPECIAL), 32)  # S1 + S2 (8 + 24)
        self.assertEqual(len(ALL_CHARS), 58)  # S1 + S2 + L (8 + 24 + 26)
    
    def test_character_set_contents(self):
        """Test that character sets contain expected characters."""
        # Test S1 structural symbols
        for char in "{}[]()<>":
            self.assertIn(char, S1)
        
        # Test S2 special characters
        for char in "!@#$%":
            self.assertIn(char, S2)
        
        # Test L lowercase letters
        for char in "abcxyz":
            self.assertIn(char, L)
    
    def test_calculate_special_char_ratio(self):
        """Test special character ratio calculation."""
        # All special characters
        self.assertEqual(calculate_special_char_ratio("{}[]"), 1.0)
        
        # No special characters
        self.assertEqual(calculate_special_char_ratio("abcdef"), 0.0)
        
        # Mixed content
        self.assertEqual(calculate_special_char_ratio("a{b}c"), 0.4)
        
        # Empty string
        self.assertEqual(calculate_special_char_ratio(""), 0.0)
    
    def test_calculate_char_distribution(self):
        """Test character distribution calculation."""
        # Test with known distribution
        dist = calculate_char_distribution("{abc}")
        self.assertEqual(dist["S1"], 0.4)  # 2/5
        self.assertEqual(dist["L"], 0.6)   # 3/5
        self.assertEqual(dist["S2"], 0.0)
        self.assertEqual(dist["other"], 0.0)
        
        # Test with all categories
        dist = calculate_char_distribution("{a!1")
        self.assertEqual(dist["S1"], 0.25)   # 1/4
        self.assertEqual(dist["L"], 0.25)    # 1/4
        self.assertEqual(dist["S2"], 0.25)   # 1/4
        self.assertEqual(dist["other"], 0.25) # 1/4
        
        # Test empty string
        dist = calculate_char_distribution("")
        self.assertEqual(dist["S1"], 0.0)
        self.assertEqual(dist["L"], 0.0)
        self.assertEqual(dist["S2"], 0.0)
        self.assertEqual(dist["other"], 0.0)
    
    def test_contains_sca_pattern(self):
        """Test SCA pattern detection."""
        # High special character content
        self.assertTrue(contains_sca_pattern("{[(!@#)]}", min_special_ratio=0.15))
        
        # Low special character content
        self.assertFalse(contains_sca_pattern("hello world", min_special_ratio=0.15))
        
        # Edge case at threshold
        self.assertTrue(contains_sca_pattern("abc{}", min_special_ratio=0.4))  # 2/5 = 0.4
        self.assertFalse(contains_sca_pattern("abc{}", min_special_ratio=0.5))


class TestCharacterSetIntegration(unittest.TestCase):
    """Integration tests for character set utilities."""
    
    def test_real_sca_sequences(self):
        """Test with realistic SCA sequences."""
        # INSET1 style - single character repeated
        seq = "{" * 100
        self.assertEqual(calculate_special_char_ratio(seq), 1.0)
        self.assertTrue(contains_sca_pattern(seq))
        
        # CROSS1 style - mixed characters
        seq = "{a!b[c#d]e}"
        ratio = calculate_special_char_ratio(seq)
        self.assertGreater(ratio, 0.5)
        self.assertTrue(contains_sca_pattern(seq))
        
        # Non-SCA content
        seq = "This is a normal sentence with few special chars."
        ratio = calculate_special_char_ratio(seq)
        self.assertLess(ratio, 0.1)
        self.assertFalse(contains_sca_pattern(seq))


if __name__ == '__main__':
    unittest.main()