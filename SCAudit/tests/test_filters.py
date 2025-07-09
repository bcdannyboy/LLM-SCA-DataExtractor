"""
Unit tests for filter pipeline.
"""

import unittest
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.core.filters import (
    LengthFilter, SpecialCharRatioFilter, EntropyFilter,
    DuplicateFilter, PatternLoopFilter, FilterPipeline
)
from SCAudit.models.data_models import Response


class TestIndividualFilters(unittest.TestCase):
    """Test individual filter implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_response = Response(
            sequence_id="test-seq-1",
            model="test-model",
            content="Test response content"
        )
    
    def test_length_filter(self):
        """Test length filter."""
        filter = LengthFilter(min_length=10)
        
        # Pass: content longer than minimum
        response = Response(content="This is a long enough response")
        passes, score = filter.check(response)
        self.assertTrue(passes)
        self.assertEqual(score, len(response.content))
        
        # Fail: content too short
        response = Response(content="Short")
        passes, score = filter.check(response)
        self.assertFalse(passes)
        self.assertEqual(score, 5)
    
    def test_special_char_ratio_filter(self):
        """Test special character ratio filter."""
        filter = SpecialCharRatioFilter(min_ratio=0.5)
        
        # Pass: high special character ratio
        response = Response(content="{[()]}")
        passes, score = filter.check(response)
        self.assertTrue(passes)
        self.assertEqual(score, 1.0)
        
        # Fail: low special character ratio
        response = Response(content="mostly text")
        passes, score = filter.check(response)
        self.assertFalse(passes)
        self.assertLess(score, 0.5)
    
    def test_entropy_filter(self):
        """Test entropy filter."""
        filter = EntropyFilter(min_entropy=1.0)
        
        # Pass: high entropy
        response = Response(content="abcdefghijk")
        passes, score = filter.check(response)
        self.assertTrue(passes)
        self.assertGreater(score, 1.0)
        
        # Fail: low entropy
        response = Response(content="aaaaaaaaaa")
        passes, score = filter.check(response)
        self.assertFalse(passes)
        self.assertEqual(score, 0.0)
    
    def test_duplicate_filter(self):
        """Test duplicate filter."""
        filter = DuplicateFilter()
        
        # First occurrence should pass
        response1 = Response(content="Unique content")
        passes, score = filter.check(response1)
        self.assertTrue(passes)
        self.assertEqual(score, 1.0)
        
        # Duplicate should fail
        response2 = Response(content="Unique content")
        passes, score = filter.check(response2)
        self.assertFalse(passes)
        self.assertEqual(score, 0.0)
        
        # Different content should pass
        response3 = Response(content="Different content")
        passes, score = filter.check(response3)
        self.assertTrue(passes)
        self.assertEqual(score, 1.0)
        
        # Test clear method
        filter.clear()
        response4 = Response(content="Unique content")
        passes, score = filter.check(response4)
        self.assertTrue(passes)
    
    def test_pattern_loop_filter(self):
        """Test pattern loop filter."""
        filter = PatternLoopFilter(max_pattern_length=10)
        
        # Pass: no pattern loops
        response = Response(content="This is normal text without patterns")
        passes, score = filter.check(response)
        self.assertTrue(passes)
        self.assertEqual(score, 1.0)
        
        # Fail: obvious pattern loop
        response = Response(content="hahahahahahaha")
        passes, score = filter.check(response)
        self.assertFalse(passes)
        self.assertLess(score, 0.5)
        
        # Fail: longer pattern
        response = Response(content="abcabcabcabc")
        passes, score = filter.check(response)
        self.assertFalse(passes)


class TestFilterPipeline(unittest.TestCase):
    """Test filter pipeline orchestration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = FilterPipeline()
    
    def test_default_pipeline(self):
        """Test pipeline with default filters."""
        # Should pass all filters
        good_response = Response(
            content="{user: 'john', data: [1,2,3]} - Complex response with variety"
        )
        self.assertTrue(self.pipeline.passes(good_response))
        
        # Should fail length filter
        short_response = Response(content="Too short")
        self.assertFalse(self.pipeline.passes(short_response))
        
        # Should fail entropy filter
        low_entropy_response = Response(content="a" * 100)
        self.assertFalse(self.pipeline.passes(low_entropy_response))
    
    def test_pipeline_evaluation(self):
        """Test detailed pipeline evaluation."""
        response = Response(content="Test content {}")
        result = self.pipeline.evaluate(response)
        
        # Check result structure
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.failed_filters, list)
        self.assertIsInstance(result.scores, dict)
        
        # Check all filters were evaluated
        expected_filters = [
            "length_filter",
            "special_char_filter",
            "entropy_filter",
            "duplicate_filter",
            "pattern_loop_filter",
            "language_coherence_filter",
            "structural_pattern_filter",
            "key_value_pair_filter",
            "data_leakage_indicator_filter",
            "contextual_anomaly_filter",
            "memorization_pattern_filter",
            "ngram_repetition_filter",
            "special_char_distribution_filter",
            "semantic_coherence_filter",
            "url_density_filter"
        ]
        for filter_name in expected_filters:
            self.assertIn(filter_name, result.scores)
    
    def test_custom_pipeline(self):
        """Test pipeline with custom filters."""
        # Create pipeline with only length filter
        pipeline = FilterPipeline([LengthFilter(min_length=5)])
        
        # Should pass
        response = Response(content="Long enough")
        self.assertTrue(pipeline.passes(response))
        
        # Should fail
        response = Response(content="No")
        self.assertFalse(pipeline.passes(response))
    
    def test_filter_management(self):
        """Test adding and removing filters."""
        pipeline = FilterPipeline([])
        
        # Start with no filters - everything passes
        response = Response(content="a")
        self.assertTrue(pipeline.passes(response))
        
        # Add length filter
        pipeline.add_filter(LengthFilter(min_length=10))
        self.assertFalse(pipeline.passes(response))
        
        # Remove the filter
        pipeline.remove_filter("length_filter")
        self.assertTrue(pipeline.passes(response))
    
    def test_batch_evaluation(self):
        """Test batch response evaluation."""
        responses = [
            Response(content="Good response with special chars: {}[]!@#$ and variety"),
            Response(content="a" * 5),  # Too short
            Response(content="x" * 100),  # Low entropy
        ]
        
        results = self.pipeline.batch_evaluate(responses)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(results[0].passed)
        self.assertFalse(results[1].passed)
        self.assertFalse(results[2].passed)
    
    def test_statistics(self):
        """Test filter statistics generation."""
        responses = [
            Response(content="Good response with special chars {}[]!@#$ and variety"),
            Response(content="Short"),
            Response(content="a" * 100),
            Response(content="Good response with special chars {}[]!@#$ and variety"),  # Duplicate
        ]
        
        stats = self.pipeline.get_statistics(responses)
        
        self.assertEqual(stats["total_responses"], 4)
        self.assertEqual(stats["passed"], 1)
        self.assertGreater(stats["pass_rate"], 0)
        self.assertLess(stats["pass_rate"], 1)
        
        # Check failure counts
        self.assertIn("length_filter", stats["filter_failure_counts"])
        self.assertIn("entropy_filter", stats["filter_failure_counts"])
        self.assertIn("duplicate_filter", stats["filter_failure_counts"])
        
        # Check average scores
        self.assertIn("length_filter", stats["average_scores"])
        self.assertIsInstance(stats["average_scores"]["length_filter"], float)


class TestFilterIntegration(unittest.TestCase):
    """Integration tests for filter pipeline."""
    
    def test_realistic_sca_responses(self):
        """Test with realistic SCA response scenarios."""
        pipeline = FilterPipeline()
        
        # Leaked training data - should pass
        leak_responses = [
            Response(content='{"name": "John Doe", "email": "john@example.com", "ssn": "123-45-6789"}'),
            Response(content='def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)'),
            Response(content='Data: {}[]()<>!@#$ with special chars mixed throughout'),
        ]
        
        for response in leak_responses:
            result = pipeline.evaluate(response)
            self.assertTrue(result.passed, f"Expected pass but failed: {result.failed_filters}")
        
        # Non-leak responses - should fail
        non_leak_responses = [
            Response(content="..." * 50),  # Low entropy
            Response(content="short"),  # Too short
            Response(content="haha" * 25),  # Pattern loop
            Response(content="just plain text without special characters" * 3),  # Low special char ratio
        ]
        
        for response in non_leak_responses:
            result = pipeline.evaluate(response)
            self.assertFalse(result.passed)


if __name__ == '__main__':
    unittest.main()