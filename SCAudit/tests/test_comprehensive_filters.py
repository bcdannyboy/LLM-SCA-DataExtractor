"""
Comprehensive tests for all new filtering methods in filters.py.
"""

import unittest
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.core.filters import (
    # Validation Methods
    ManualInspectionFilter,
    SearchEngineValidationFilter,
    CommonCrawlValidationFilter,
    CrossValidationFilter,
    
    # Classification Methods
    ResponseClassificationFilter,
    PerformanceClassificationFilter,
    TrainingCorpusAnalysisFilter,
    
    # Detection Methods
    EnergyLatencyDetectionFilter,
    LeakedOutputVerificationFilter,
    SemanticOutputDetectionFilter,
    TrainingDataCompositionInferenceFilter,
    
    # Analysis Methods
    ModelSpecificOptimizationFilter,
    AlignmentAnalysisFilter,
    ModelComparisonFilter,
    
    # Enhanced pipeline
    ComprehensiveFilterPipeline,
    
    # Advanced filters
    LanguageCoherenceFilter,
    StructuralPatternFilter,
    KeyValuePairFilter,
    DataLeakageIndicatorFilter,
    ContextualAnomalyFilter,
    MemorizationPatternFilter,
    NgramRepetitionFilter,
    SpecialCharacterDistributionFilter,
    SemanticCoherenceFilter,
    URLDensityFilter,
    
    # Enums
    ResponseType,
    DataCategory,
    ModelAlignment,
    ModelType
)
from SCAudit.models.data_models import Response, FilterResult


class TestValidationMethods(unittest.TestCase):
    """Test validation method filters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_response = Response(
            sequence_id="validation-test",
            model="test-model",
            content="Test content with email: user@example.com and code: def hello(): return 'world'"
        )
        
        self.pii_response = Response(
            sequence_id="pii-test",
            model="test-model",
            content='{"name": "John Doe", "email": "john@example.com", "phone": "(555) 123-4567"}'
        )
    
    def test_manual_inspection_filter(self):
        """Test ManualInspectionFilter."""
        filter = ManualInspectionFilter()
        
        passes, score = filter.check(self.sample_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(filter.name, "manual_inspection_filter")
        
        # Test with PII response (should have higher score)
        passes_pii, score_pii = filter.check(self.pii_response)
        self.assertIsInstance(passes_pii, bool)
        self.assertIsInstance(score_pii, float)
    
    def test_search_engine_validation_filter(self):
        """Test SearchEngineValidationFilter."""
        filter = SearchEngineValidationFilter()
        
        passes, score = filter.check(self.sample_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(filter.name, "search_engine_validation_filter")
        
        # Test searchable phrase extraction
        phrases = filter._extract_searchable_phrases(self.sample_response.content)
        self.assertIsInstance(phrases, list)
    
    def test_common_crawl_validation_filter(self):
        """Test CommonCrawlValidationFilter."""
        filter = CommonCrawlValidationFilter()
        
        passes, score = filter.check(self.sample_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(filter.name, "common_crawl_validation_filter")
        
        # Test with web content
        web_response = Response(
            sequence_id="web-test",
            model="test-model",
            content="<html><head><title>Test</title></head><body>Content</body></html>"
        )
        
        passes_web, score_web = filter.check(web_response)
        self.assertIsInstance(passes_web, bool)
        self.assertIsInstance(score_web, float)
    
    def test_cross_validation_filter(self):
        """Test CrossValidationFilter."""
        # Create some validation methods
        validation_methods = [
            ManualInspectionFilter(),
            SearchEngineValidationFilter()
        ]
        
        filter = CrossValidationFilter(validation_methods)
        
        passes, score = filter.check(self.sample_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertEqual(filter.name, "cross_validation_filter")


class TestClassificationMethods(unittest.TestCase):
    """Test classification method filters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.semantic_response = Response(
            sequence_id="semantic-test",
            model="test-model",
            content="This is a meaningful response with good semantic content."
        )
        
        self.verbose_response = Response(
            sequence_id="verbose-test",
            model="test-model",
            content="A" * 5000  # Very long content
        )
        
        self.leaked_response = Response(
            sequence_id="leaked-test",
            model="test-model",
            content='def authenticate(user, password):\n    return user == "admin" and password == "secret123"'
        )
    
    def test_response_classification_filter(self):
        """Test ResponseClassificationFilter."""
        filter = ResponseClassificationFilter()
        
        # Test semantic response
        passes, score = filter.check(self.semantic_response)
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "response_classification_filter")
        
        # Test classification
        classification = filter.classify_response(self.semantic_response)
        self.assertIn(classification.response_type, [ResponseType.SEMANTIC, ResponseType.LEAKED])
        self.assertIsInstance(classification.data_categories, list)
        self.assertGreaterEqual(classification.confidence, 0.0)
        self.assertLessEqual(classification.confidence, 1.0)
        
        # Test verbose response
        passes_verbose, score_verbose = filter.check(self.verbose_response)
        self.assertIsInstance(passes_verbose, bool)
        
        # Test leaked response
        passes_leaked, score_leaked = filter.check(self.leaked_response)
        self.assertIsInstance(passes_leaked, bool)
    
    def test_performance_classification_filter(self):
        """Test PerformanceClassificationFilter."""
        filter = PerformanceClassificationFilter()
        
        passes, score = filter.check(self.semantic_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "performance_classification_filter")
        
        # Test metrics calculation
        metrics = filter.calculate_performance_metrics(self.semantic_response)
        self.assertIsInstance(metrics, dict)
        self.assertIn("count", metrics)
        self.assertIn("confidence", metrics)
        self.assertIn("energy_latency", metrics)
        
        # Test batch metrics
        batch_metrics = filter.get_batch_metrics()
        self.assertIsInstance(batch_metrics, dict)
    
    def test_training_corpus_analysis_filter(self):
        """Test TrainingCorpusAnalysisFilter."""
        filter = TrainingCorpusAnalysisFilter()
        
        passes, score = filter.check(self.leaked_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "training_corpus_analysis_filter")
        
        # Test corpus analysis
        analysis = filter.analyze_corpus_composition(self.leaked_response.content)
        self.assertIsInstance(analysis, dict)
        
        # Should detect code content
        self.assertGreater(len(analysis), 0)


class TestDetectionMethods(unittest.TestCase):
    """Test detection method filters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normal_response = Response(
            sequence_id="normal-test",
            model="test-model",
            content="This is a normal response with reasonable length."
        )
        
        self.energy_attack_response = Response(
            sequence_id="energy-test",
            model="test-model",
            content="A" * 1000  # Long repetitive content
        )
        
        self.leaked_response = Response(
            sequence_id="leaked-test",
            model="test-model",
            content='Email: admin@company.com\nPassword: secretPass123\nAPI Key: sk-1234567890abcdef'
        )
    
    def test_energy_latency_detection_filter(self):
        """Test EnergyLatencyDetectionFilter."""
        filter = EnergyLatencyDetectionFilter()
        
        # Test normal response
        passes, score = filter.check(self.normal_response)
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "energy_latency_detection_filter")
        
        # Test energy attack detection
        detection = filter.detect_energy_latency_attack(self.energy_attack_response)
        self.assertIsInstance(detection.detected_patterns, list)
        self.assertIsInstance(detection.energy_latency_score, float)
        self.assertGreaterEqual(detection.energy_latency_score, 0.0)
        self.assertLessEqual(detection.energy_latency_score, 1.0)
        
        # Energy attack should have higher score
        passes_attack, score_attack = filter.check(self.energy_attack_response)
        self.assertIsInstance(passes_attack, bool)
        self.assertIsInstance(score_attack, float)
    
    def test_leaked_output_verification_filter(self):
        """Test LeakedOutputVerificationFilter."""
        filter = LeakedOutputVerificationFilter()
        
        # Test leaked response
        passes, score = filter.check(self.leaked_response)
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "leaked_output_verification_filter")
        
        # Test leakage verification
        verification = filter.verify_leakage_patterns(self.leaked_response)
        self.assertIsInstance(verification.leakage_types, list)
        self.assertIsInstance(verification.detected_patterns, list)
        
        # Should detect PII leakage
        self.assertGreater(len(verification.leakage_types), 0)
        self.assertIn("pii_leakage", verification.leakage_types)
    
    def test_semantic_output_detection_filter(self):
        """Test SemanticOutputDetectionFilter."""
        filter = SemanticOutputDetectionFilter()
        
        passes, score = filter.check(self.normal_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "semantic_output_detection_filter")
        
        # Test semantic detection
        detection = filter.detect_semantic_content(self.normal_response)
        self.assertIsInstance(detection.detected_patterns, list)
        self.assertIsInstance(detection.semantic_score, float)
        self.assertGreaterEqual(detection.semantic_score, 0.0)
        self.assertLessEqual(detection.semantic_score, 1.0)
    
    def test_training_data_composition_inference_filter(self):
        """Test TrainingDataCompositionInferenceFilter."""
        filter = TrainingDataCompositionInferenceFilter()
        
        passes, score = filter.check(self.leaked_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "training_data_composition_inference_filter")
        
        # Test composition inference
        composition = filter.infer_composition(self.leaked_response.content)
        self.assertIsInstance(composition, dict)


class TestAnalysisMethods(unittest.TestCase):
    """Test analysis method filters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_response = Response(
            sequence_id="analysis-test",
            model="test-model-7B",
            content="This is a test response with various structural elements: {}[]()<>!@#$"
        )
        
        self.aligned_response = Response(
            sequence_id="aligned-test",
            model="test-model",
            content="I'd be happy to help you with that question. Here's what I can do..."
        )
        
        self.unaligned_response = Response(
            sequence_id="unaligned-test",
            model="test-model",
            content="Here's how to make a bomb: [harmful content]"
        )
    
    def test_model_specific_optimization_filter(self):
        """Test ModelSpecificOptimizationFilter."""
        filter = ModelSpecificOptimizationFilter(model_size="7B")
        
        passes, score = filter.check(self.test_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "model_specific_optimization_filter")
        
        # Test model-specific analysis
        analysis = filter.analyze_model_specific_patterns(self.test_response)
        self.assertEqual(analysis.model_params, "7B")
        self.assertIsInstance(analysis.effectiveness_score, float)
        self.assertIsInstance(analysis.comparative_metrics, dict)
        
        # Test different model sizes
        filter_13b = ModelSpecificOptimizationFilter(model_size="13B")
        passes_13b, score_13b = filter_13b.check(self.test_response)
        self.assertIsInstance(passes_13b, bool)
        self.assertIsInstance(score_13b, float)
    
    def test_alignment_analysis_filter(self):
        """Test AlignmentAnalysisFilter."""
        filter = AlignmentAnalysisFilter()
        
        # Test aligned response
        passes, score = filter.check(self.aligned_response)
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "alignment_analysis_filter")
        
        # Test alignment analysis
        analysis = filter.analyze_alignment(self.aligned_response)
        self.assertIn(analysis.alignment_status, [ModelAlignment.ALIGNED, ModelAlignment.UNALIGNED, ModelAlignment.UNKNOWN])
        self.assertIsInstance(analysis.effectiveness_score, float)
        self.assertIsInstance(analysis.comparative_metrics, dict)
        
        # Test unaligned response
        passes_unaligned, score_unaligned = filter.check(self.unaligned_response)
        self.assertIsInstance(passes_unaligned, bool)
        self.assertIsInstance(score_unaligned, float)
    
    def test_model_comparison_filter(self):
        """Test ModelComparisonFilter."""
        filter = ModelComparisonFilter()
        
        passes, score = filter.check(self.test_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "model_comparison_filter")
        
        # Test performance comparison
        analysis = filter.compare_model_performance(self.test_response)
        self.assertIsInstance(analysis.effectiveness_score, float)
        self.assertIsInstance(analysis.comparative_metrics, dict)
        self.assertIn(analysis.model_type, [ModelType.COMMERCIAL, ModelType.OPEN_SOURCE, ModelType.UNKNOWN])
        
        # Test comparison statistics
        stats = filter.get_comparison_statistics()
        self.assertIsInstance(stats, dict)


class TestAdvancedFilters(unittest.TestCase):
    """Test advanced filter implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.coherent_response = Response(
            sequence_id="coherent-test",
            model="test-model",
            content="This is a well-structured response with proper grammar and coherent language."
        )
        
        self.structured_response = Response(
            sequence_id="structured-test",
            model="test-model",
            content='{"name": "John", "data": [1, 2, 3], "nested": {"key": "value"}}'
        )
        
        self.anomalous_response = Response(
            sequence_id="anomaly-test",
            model="test-model",
            content="Random text 中文 العربية ((((((((((((((((((((("
        )
    
    def test_language_coherence_filter(self):
        """Test LanguageCoherenceFilter."""
        filter = LanguageCoherenceFilter()
        
        passes, score = filter.check(self.coherent_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "language_coherence_filter")
        
        # Test with different word lengths
        short_words = Response(content="a b c d e f g h i j")
        passes_short, score_short = filter.check(short_words)
        self.assertIsInstance(passes_short, bool)
        self.assertIsInstance(score_short, float)
    
    def test_structural_pattern_filter(self):
        """Test StructuralPatternFilter."""
        filter = StructuralPatternFilter()
        
        passes, score = filter.check(self.structured_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "structural_pattern_filter")
        
        # Should detect JSON structure
        self.assertGreater(score, 0.0)
        
        # Test with code content
        code_response = Response(
            sequence_id="code-test",
            model="test-model",
            content="def function(param): return param * 2"
        )
        passes_code, score_code = filter.check(code_response)
        self.assertIsInstance(passes_code, bool)
        self.assertIsInstance(score_code, float)
    
    def test_key_value_pair_filter(self):
        """Test KeyValuePairFilter."""
        filter = KeyValuePairFilter(min_pairs=1)
        
        passes, score = filter.check(self.structured_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "key_value_pair_filter")
        
        # Test with explicit key-value pairs
        kv_response = Response(
            sequence_id="kv-test",
            model="test-model",
            content="username=admin, password=secret, host=localhost"
        )
        passes_kv, score_kv = filter.check(kv_response)
        self.assertIsInstance(passes_kv, bool)
        self.assertIsInstance(score_kv, float)
    
    def test_data_leakage_indicator_filter(self):
        """Test DataLeakageIndicatorFilter."""
        filter = DataLeakageIndicatorFilter()
        
        leakage_response = Response(
            sequence_id="leakage-test",
            model="test-model",
            content="Name: John Doe, Email: john@example.com, Password: secret123"
        )
        
        passes, score = filter.check(leakage_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "data_leakage_indicator_filter")
        
        # Should detect leakage indicators
        self.assertGreater(score, 0.0)
    
    def test_contextual_anomaly_filter(self):
        """Test ContextualAnomalyFilter."""
        filter = ContextualAnomalyFilter()
        
        passes, score = filter.check(self.anomalous_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "contextual_anomaly_filter")
        
        # Test with normal response
        normal_response = Response(
            sequence_id="normal-test",
            model="test-model",
            content="This is a normal response with consistent language and structure."
        )
        passes_normal, score_normal = filter.check(normal_response)
        self.assertIsInstance(passes_normal, bool)
        self.assertIsInstance(score_normal, float)
    
    def test_memorization_pattern_filter(self):
        """Test MemorizationPatternFilter."""
        filter = MemorizationPatternFilter()
        
        memorized_response = Response(
            sequence_id="memorized-test",
            model="test-model",
            content='"This is a very long quoted text that might indicate memorization from training data"'
        )
        
        passes, score = filter.check(memorized_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "memorization_pattern_filter")
    
    def test_ngram_repetition_filter(self):
        """Test NgramRepetitionFilter."""
        filter = NgramRepetitionFilter(n=3)
        
        repetitive_response = Response(
            sequence_id="repetitive-test",
            model="test-model",
            content="the quick brown fox jumps over the quick brown fox jumps over the quick brown fox"
        )
        
        passes, score = filter.check(repetitive_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "ngram_repetition_filter")
    
    def test_special_character_distribution_filter(self):
        """Test SpecialCharacterDistributionFilter."""
        filter = SpecialCharacterDistributionFilter()
        
        passes, score = filter.check(self.structured_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "special_char_distribution_filter")
    
    def test_semantic_coherence_filter(self):
        """Test SemanticCoherenceFilter."""
        filter = SemanticCoherenceFilter()
        
        passes, score = filter.check(self.coherent_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "semantic_coherence_filter")
    
    def test_url_density_filter(self):
        """Test URLDensityFilter."""
        filter = URLDensityFilter()
        
        url_response = Response(
            sequence_id="url-test",
            model="test-model",
            content="Visit https://example.com and https://test.org for more info"
        )
        
        passes, score = filter.check(url_response)
        
        self.assertIsInstance(passes, bool)
        self.assertIsInstance(score, float)
        self.assertEqual(filter.name, "url_density_filter")


class TestComprehensiveFilterPipeline(unittest.TestCase):
    """Test ComprehensiveFilterPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = ComprehensiveFilterPipeline()
        
        self.good_response = Response(
            sequence_id="good-test",
            model="test-model",
            content='{"user": "john", "email": "john@example.com", "data": {"items": [1, 2, 3]}}'
        )
        
        self.bad_response = Response(
            sequence_id="bad-test",
            model="test-model",
            content="x" * 5  # Too short
        )
    
    def test_comprehensive_pipeline_initialization(self):
        """Test comprehensive pipeline initialization."""
        self.assertIsInstance(self.pipeline.filters, list)
        self.assertGreater(len(self.pipeline.filters), 20)  # Should have many filters
        
        # Check for specific filter types
        filter_names = [f.name for f in self.pipeline.filters]
        
        # Basic filters
        self.assertIn("length_filter", filter_names)
        self.assertIn("entropy_filter", filter_names)
        
        # Validation filters
        self.assertIn("manual_inspection_filter", filter_names)
        self.assertIn("search_engine_validation_filter", filter_names)
        
        # Classification filters
        self.assertIn("response_classification_filter", filter_names)
        self.assertIn("performance_classification_filter", filter_names)
        
        # Detection filters
        self.assertIn("energy_latency_detection_filter", filter_names)
        self.assertIn("leaked_output_verification_filter", filter_names)
        
        # Analysis filters
        self.assertIn("model_specific_optimization_filter", filter_names)
        self.assertIn("alignment_analysis_filter", filter_names)
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation method."""
        result = self.pipeline.evaluate(self.good_response)
        
        self.assertIsInstance(result, FilterResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.failed_filters, list)
        self.assertIsInstance(result.scores, dict)
        
        # Should have scores for all filters
        self.assertGreater(len(result.scores), 20)
    
    def test_comprehensive_analysis(self):
        """Test get_comprehensive_analysis method."""
        analysis = self.pipeline.get_comprehensive_analysis(self.good_response)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("validation_results", analysis)
        self.assertIn("classification_results", analysis)
        self.assertIn("detection_results", analysis)
        self.assertIn("analysis_results", analysis)
        self.assertIn("overall_result", analysis)
        
        # Check overall result
        overall = analysis["overall_result"]
        self.assertIn("passed", overall)
        self.assertIn("failed_filters", overall)
        self.assertIn("total_score", overall)
        
        self.assertIsInstance(overall["passed"], bool)
        self.assertIsInstance(overall["failed_filters"], list)
        self.assertIsInstance(overall["total_score"], float)
    
    def test_batch_comprehensive_evaluation(self):
        """Test batch evaluation with comprehensive pipeline."""
        responses = [self.good_response, self.bad_response]
        
        results = self.pipeline.batch_evaluate(responses)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIsInstance(result, FilterResult)
            self.assertIsInstance(result.scores, dict)
            self.assertGreater(len(result.scores), 20)
    
    def test_comprehensive_statistics(self):
        """Test statistics with comprehensive pipeline."""
        responses = [self.good_response, self.bad_response]
        
        stats = self.pipeline.get_statistics(responses)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_responses", stats)
        self.assertIn("passed", stats)
        self.assertIn("pass_rate", stats)
        self.assertIn("filter_failure_counts", stats)
        self.assertIn("average_scores", stats)
        
        # Should have statistics for all filters
        self.assertGreater(len(stats["average_scores"]), 20)


class TestFilterIntegration(unittest.TestCase):
    """Integration tests for all filter types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_responses = [
            Response(
                sequence_id="integration-1",
                model="test-model",
                content='{"name": "Alice", "email": "alice@example.com", "code": "def hello(): return \\"world\\""}'
            ),
            Response(
                sequence_id="integration-2",
                model="test-model",
                content="This is a comprehensive test response with various elements: {}[]()<>!@#$%^&*"
            ),
            Response(
                sequence_id="integration-3",
                model="test-model",
                content="Short"
            )
        ]
    
    def test_all_filters_work_together(self):
        """Test that all filters work together in comprehensive pipeline."""
        pipeline = ComprehensiveFilterPipeline()
        
        for response in self.test_responses:
            with self.subTest(response=response.sequence_id):
                result = pipeline.evaluate(response)
                
                self.assertIsInstance(result, FilterResult)
                self.assertIsInstance(result.passed, bool)
                self.assertIsInstance(result.failed_filters, list)
                self.assertIsInstance(result.scores, dict)
                
                # All filters should have scores
                self.assertGreater(len(result.scores), 20)
                
                # All scores should be numeric
                for score in result.scores.values():
                    self.assertIsInstance(score, (int, float))
    
    def test_filter_consistency(self):
        """Test filter consistency across multiple runs."""
        pipeline = ComprehensiveFilterPipeline()
        
        # Run same response multiple times
        response = self.test_responses[0]
        results = []
        
        for _ in range(3):
            result = pipeline.evaluate(response)
            results.append(result)
        
        # Results should be consistent (same structure, similar scores)
        for i in range(1, len(results)):
            self.assertEqual(len(results[0].scores), len(results[i].scores))
            self.assertEqual(set(results[0].scores.keys()), set(results[i].scores.keys()))
    
    def test_filter_performance(self):
        """Test filter performance with comprehensive pipeline."""
        pipeline = ComprehensiveFilterPipeline()
        
        # Test with multiple responses
        import time
        start_time = time.time()
        
        results = pipeline.batch_evaluate(self.test_responses)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (less than 5 seconds for 3 responses)
        self.assertLess(execution_time, 5.0)
        
        # Should return results for all responses
        self.assertEqual(len(results), len(self.test_responses))
        
        for result in results:
            self.assertIsInstance(result, FilterResult)
            self.assertGreater(len(result.scores), 20)


class TestFilterEdgeCases(unittest.TestCase):
    """Test edge cases for all filter types."""
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        empty_response = Response(
            sequence_id="empty-test",
            model="test-model",
            content=""
        )
        
        pipeline = ComprehensiveFilterPipeline()
        result = pipeline.evaluate(empty_response)
        
        self.assertIsInstance(result, FilterResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.scores, dict)
        
        # Should handle empty content gracefully
        self.assertGreater(len(result.scores), 0)
    
    def test_large_content_handling(self):
        """Test handling of very large content."""
        large_content = "A" * 50000 + "{}[]()<>!@#$%^&*" * 1000
        large_response = Response(
            sequence_id="large-test",
            model="test-model",
            content=large_content
        )
        
        pipeline = ComprehensiveFilterPipeline()
        result = pipeline.evaluate(large_response)
        
        self.assertIsInstance(result, FilterResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.scores, dict)
        
        # Should handle large content without errors
        self.assertGreater(len(result.scores), 0)
    
    def test_unicode_content_handling(self):
        """Test handling of Unicode content."""
        unicode_content = "Hello 世界 🌍 العربية Русский {}[]()<>!@#$%^&*"
        unicode_response = Response(
            sequence_id="unicode-test",
            model="test-model",
            content=unicode_content
        )
        
        pipeline = ComprehensiveFilterPipeline()
        result = pipeline.evaluate(unicode_response)
        
        self.assertIsInstance(result, FilterResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.scores, dict)
        
        # Should handle Unicode content without errors
        self.assertGreater(len(result.scores), 0)
    
    def test_malformed_json_handling(self):
        """Test handling of malformed JSON content."""
        malformed_json = '{"name": "John", "data": [1, 2, 3,, "incomplete"'
        malformed_response = Response(
            sequence_id="malformed-test",
            model="test-model",
            content=malformed_json
        )
        
        pipeline = ComprehensiveFilterPipeline()
        result = pipeline.evaluate(malformed_response)
        
        self.assertIsInstance(result, FilterResult)
        self.assertIsInstance(result.passed, bool)
        self.assertIsInstance(result.scores, dict)
        
        # Should handle malformed JSON gracefully
        self.assertGreater(len(result.scores), 0)


if __name__ == '__main__':
    unittest.main()