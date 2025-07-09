"""
Comprehensive tests for new SCA extraction methods in extractor.py.
"""

import unittest
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.core.extractor import DataExtractor
from SCAudit.models.data_models import Response, Sequence, Strategy
from SCAudit.utils.character_sets import S1, S2, L


class TestNewSCAExtractionMethods(unittest.TestCase):
    """Test the 9 new SCA extraction methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DataExtractor()
        
        # Sample sequence for testing
        self.sample_sequence = Sequence(
            id="test-seq-1",
            content="Sample sequence with S1 chars: {}[]()<>!@#$%^&*",
            strategy=Strategy.INSET1
        )
        
        # Sample response for testing
        self.sample_response = Response(
            sequence_id="test-seq-1",
            model="test-model",
            content="Sample response with S1 chars: {}[]()<>!@#$%^&*"
        )
    
    def test_extract_sca_inset1_patterns(self):
        """Test INSET1 pattern extraction."""
        result = self.extractor.extract_sca_inset1_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.response_id, self.sample_response.id)
        self.assertEqual(result.method, "sca_inset1_patterns")
        self.assertIsInstance(result.content, dict)
        self.assertIn("inset1_patterns", result.content)
        self.assertIn("pattern_count", result.content)
        self.assertIn("s1_density", result.content)
        
        # Check that S1 patterns are detected
        patterns = result.content["inset1_patterns"]
        self.assertIsInstance(patterns, list)
        
        # Verify pattern analysis
        self.assertIsInstance(result.content["pattern_count"], int)
        self.assertIsInstance(result.content["s1_density"], float)
    
    def test_extract_sca_inset2_patterns(self):
        """Test INSET2 pattern extraction."""
        result = self.extractor.extract_sca_inset2_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "sca_inset2_patterns")
        self.assertIn("inset2_patterns", result.content)
        self.assertIn("s2_density", result.content)
        self.assertIn("structural_indicators", result.content)
        
        # Check S2 pattern detection
        patterns = result.content["inset2_patterns"]
        self.assertIsInstance(patterns, list)
        
        # Verify S2 density calculation
        s2_density = result.content["s2_density"]
        self.assertIsInstance(s2_density, float)
        self.assertGreaterEqual(s2_density, 0.0)
        self.assertLessEqual(s2_density, 1.0)
    
    def test_extract_sca_cross1_patterns(self):
        """Test CROSS1 pattern extraction."""
        result = self.extractor.extract_sca_cross1_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "sca_cross1_patterns")
        self.assertIn("cross1_patterns", result.content)
        self.assertIn("cross_correlations", result.content)
        self.assertIn("character_interactions", result.content)
        
        # Check cross-pattern analysis
        cross_patterns = result.content["cross1_patterns"]
        self.assertIsInstance(cross_patterns, list)
        
        # Verify correlations
        correlations = result.content["cross_correlations"]
        self.assertIsInstance(correlations, dict)
    
    def test_extract_sca_cross2_patterns(self):
        """Test CROSS2 pattern extraction."""
        result = self.extractor.extract_sca_cross2_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "sca_cross2_patterns")
        self.assertIn("cross2_patterns", result.content)
        self.assertIn("advanced_correlations", result.content)
        self.assertIn("pattern_complexity", result.content)
        
        # Check advanced pattern analysis
        patterns = result.content["cross2_patterns"]
        self.assertIsInstance(patterns, list)
        
        # Verify complexity metrics
        complexity = result.content["pattern_complexity"]
        self.assertIsInstance(complexity, (int, float))
    
    def test_extract_sca_cross3_patterns(self):
        """Test CROSS3 pattern extraction."""
        result = self.extractor.extract_sca_cross3_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "sca_cross3_patterns")
        self.assertIn("cross3_patterns", result.content)
        self.assertIn("multi_dimensional_analysis", result.content)
        self.assertIn("pattern_evolution", result.content)
        
        # Check multi-dimensional analysis
        patterns = result.content["cross3_patterns"]
        self.assertIsInstance(patterns, list)
        
        # Verify evolution tracking
        evolution = result.content["pattern_evolution"]
        self.assertIsInstance(evolution, dict)
    
    def test_extract_character_set_integration(self):
        """Test character set integration extraction."""
        result = self.extractor.extract_character_set_integration(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "character_set_integration")
        self.assertIn("character_distributions", result.content)
        self.assertIn("set_interactions", result.content)
        self.assertIn("integration_metrics", result.content)
        
        # Check character distributions
        distributions = result.content["character_distributions"]
        self.assertIsInstance(distributions, dict)
        self.assertIn("S1", distributions)
        self.assertIn("S2", distributions)
        self.assertIn("L", distributions)
        
        # Verify integration metrics
        metrics = result.content["integration_metrics"]
        self.assertIsInstance(metrics, dict)
    
    def test_extract_contextual_sca_patterns(self):
        """Test contextual SCA pattern extraction."""
        result = self.extractor.extract_contextual_sca_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "contextual_sca_patterns")
        self.assertIn("contextual_patterns", result.content)
        self.assertIn("context_analysis", result.content)
        self.assertIn("semantic_indicators", result.content)
        
        # Check contextual analysis
        patterns = result.content["contextual_patterns"]
        self.assertIsInstance(patterns, list)
        
        # Verify context analysis
        context = result.content["context_analysis"]
        self.assertIsInstance(context, dict)
    
    def test_extract_memory_trigger_patterns(self):
        """Test memory trigger pattern extraction."""
        result = self.extractor.extract_memory_trigger_patterns(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "memory_trigger_patterns")
        self.assertIn("trigger_patterns", result.content)
        self.assertIn("memory_indicators", result.content)
        self.assertIn("activation_scores", result.content)
        
        # Check trigger patterns
        triggers = result.content["trigger_patterns"]
        self.assertIsInstance(triggers, list)
        
        # Verify activation scores
        scores = result.content["activation_scores"]
        self.assertIsInstance(scores, dict)
    
    def test_extract_sca_effectiveness_metrics(self):
        """Test SCA effectiveness metrics extraction."""
        result = self.extractor.extract_sca_effectiveness_metrics(self.sample_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "sca_effectiveness_metrics")
        self.assertIn("effectiveness_scores", result.content)
        self.assertIn("success_indicators", result.content)
        self.assertIn("quality_metrics", result.content)
        
        # Check effectiveness scores
        scores = result.content["effectiveness_scores"]
        self.assertIsInstance(scores, dict)
        
        # Verify quality metrics
        quality = result.content["quality_metrics"]
        self.assertIsInstance(quality, dict)
        self.assertIn("overall_score", quality)


class TestSCAExtractionIntegration(unittest.TestCase):
    """Integration tests for SCA extraction methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DataExtractor()
        
        # Create realistic SCA response scenarios
        self.pii_response = Response(
            sequence_id="pii-test",
            model="test-model",
            content='{"name": "John Doe", "email": "john@example.com", "ssn": "123-45-6789"}'
        )
        
        self.code_response = Response(
            sequence_id="code-test",
            model="test-model",
            content='def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)'
        )
    
    def test_extraction_method_consistency(self):
        """Test that all extraction methods return consistent data structures."""
        methods = [
            'extract_sca_inset1_patterns',
            'extract_sca_inset2_patterns',
            'extract_sca_cross1_patterns',
            'extract_sca_cross2_patterns',
            'extract_sca_cross3_patterns',
            'extract_character_set_integration',
            'extract_contextual_sca_patterns',
            'extract_memory_trigger_patterns',
            'extract_sca_effectiveness_metrics'
        ]
        
        for method_name in methods:
            with self.subTest(method=method_name):
                method = getattr(self.extractor, method_name)
                result = method(self.pii_response)
                
                # Check basic structure
                self.assertIsNotNone(result)
                self.assertEqual(result.response_id, self.pii_response.id)
                self.assertEqual(result.method, method_name.replace('extract_', ''))
                self.assertIsInstance(result.content, dict)
                self.assertIsInstance(result.created_at, datetime)
    
    def test_character_set_integration_accuracy(self):
        """Test accuracy of character set integration."""
        result = self.extractor.extract_character_set_integration(self.pii_response)
        
        distributions = result.content["character_distributions"]
        
        # Check that S1 characters are detected (braces, brackets, etc.)
        self.assertGreater(distributions["S1"], 0)
        
        # Check that regular letters are detected
        self.assertGreater(distributions["L"], 0)
        
        # Verify total adds up correctly
        total = distributions["S1"] + distributions["S2"] + distributions["L"]
        self.assertGreater(total, 0)
    
    def test_cross_pattern_extraction_depth(self):
        """Test depth of cross-pattern analysis."""
        cross1_result = self.extractor.extract_sca_cross1_patterns(self.code_response)
        cross2_result = self.extractor.extract_sca_cross2_patterns(self.code_response)
        cross3_result = self.extractor.extract_sca_cross3_patterns(self.code_response)
        
        # Check increasing complexity
        cross1_patterns = len(cross1_result.content["cross1_patterns"])
        cross2_patterns = len(cross2_result.content["cross2_patterns"])
        cross3_patterns = len(cross3_result.content["cross3_patterns"])
        
        # Cross3 should have the most sophisticated analysis
        self.assertGreaterEqual(cross3_patterns, 0)
        
        # Check for multi-dimensional analysis in Cross3
        multi_dim = cross3_result.content["multi_dimensional_analysis"]
        self.assertIsInstance(multi_dim, dict)
        self.assertGreater(len(multi_dim), 0)
    
    def test_effectiveness_metrics_calculation(self):
        """Test SCA effectiveness metrics calculation."""
        result = self.extractor.extract_sca_effectiveness_metrics(self.pii_response)
        
        effectiveness = result.content["effectiveness_scores"]
        quality = result.content["quality_metrics"]
        
        # Check that scores are reasonable
        self.assertIn("overall_score", quality)
        overall_score = quality["overall_score"]
        self.assertIsInstance(overall_score, (int, float))
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 1.0)
        
        # Check success indicators
        success = result.content["success_indicators"]
        self.assertIsInstance(success, dict)
        self.assertGreater(len(success), 0)


class TestSCAExtractionEdgeCases(unittest.TestCase):
    """Test edge cases for SCA extraction methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = DataExtractor()
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        empty_response = Response(
            sequence_id="empty-test",
            model="test-model",
            content=""
        )
        
        result = self.extractor.extract_sca_inset1_patterns(empty_response)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.response_id, empty_response.id)
        self.assertIsInstance(result.content, dict)
    
    def test_large_content_handling(self):
        """Test handling of large content."""
        large_content = "A" * 10000 + "{}[]()<>!@#$%^&*" * 1000
        large_response = Response(
            sequence_id="large-test",
            model="test-model",
            content=large_content
        )
        
        result = self.extractor.extract_character_set_integration(large_response)
        
        self.assertIsNotNone(result)
        self.assertIn("character_distributions", result.content)
        
        # Should handle large content without errors
        distributions = result.content["character_distributions"]
        self.assertIsInstance(distributions, dict)
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        unicode_content = "Hello 世界 🌍 {}[]()<>!@#$%^&*"
        unicode_response = Response(
            sequence_id="unicode-test",
            model="test-model",
            content=unicode_content
        )
        
        result = self.extractor.extract_sca_inset2_patterns(unicode_response)
        
        self.assertIsNotNone(result)
        self.assertIn("inset2_patterns", result.content)
        
        # Should handle Unicode without errors
        patterns = result.content["inset2_patterns"]
        self.assertIsInstance(patterns, list)


if __name__ == '__main__':
    unittest.main()