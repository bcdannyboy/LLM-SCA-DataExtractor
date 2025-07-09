"""
Unit tests for data models.
"""

import unittest
from datetime import datetime, timezone
import hashlib
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from SCAudit.models.data_models import (
    Sequence, Response, Judgment, ExtractedData,
    Strategy, JudgmentVerdict, FilterResult
)


class TestDataModels(unittest.TestCase):
    """Test data model classes."""
    
    def test_sequence_creation(self):
        """Test Sequence model creation and derived fields."""
        # Basic creation
        seq = Sequence(content="test content")
        
        # Check auto-generated fields
        self.assertIsNotNone(seq.id)
        self.assertEqual(seq.length, 12)
        self.assertEqual(
            seq.sha256,
            hashlib.sha256("test content".encode()).hexdigest()
        )
        self.assertIsInstance(seq.created_at, datetime)
        
        # With strategy
        seq = Sequence(content="{{{", strategy=Strategy.INSET1)
        self.assertEqual(seq.strategy, Strategy.INSET1)
        self.assertEqual(seq.length, 3)
        
        # With metadata
        seq = Sequence(
            content="test",
            metadata={"source_file": "test.txt", "line_number": 5}
        )
        self.assertEqual(seq.metadata["source_file"], "test.txt")
        self.assertEqual(seq.metadata["line_number"], 5)
    
    def test_response_creation(self):
        """Test Response model creation."""
        resp = Response(
            sequence_id="seq-123",
            model="gpt-4",
            content="Model response",
            tokens_used=150,
            latency_ms=1234.5
        )
        
        self.assertIsNotNone(resp.id)
        self.assertEqual(resp.sequence_id, "seq-123")
        self.assertEqual(resp.model, "gpt-4")
        self.assertEqual(resp.content, "Model response")
        self.assertEqual(resp.tokens_used, 150)
        self.assertEqual(resp.latency_ms, 1234.5)
        
        # With error metadata
        error_resp = Response(
            sequence_id="seq-456",
            model="gpt-4",
            content="ERROR: Rate limit exceeded",
            metadata={"error": True, "error_type": "RateLimitError"}
        )
        self.assertTrue(error_resp.metadata["error"])
        self.assertEqual(error_resp.metadata["error_type"], "RateLimitError")
    
    def test_judgment_creation(self):
        """Test Judgment model creation."""
        # Basic judgment
        judge = Judgment(
            response_id="resp-123",
            verdict=JudgmentVerdict.LEAK,
            confidence=0.95,
            is_leak=True,
            judge_model="gpt-4"
        )
        
        self.assertEqual(judge.verdict, JudgmentVerdict.LEAK)
        self.assertEqual(judge.confidence, 0.95)
        self.assertTrue(judge.is_leak)
        
        # With ensemble votes
        judge = Judgment(
            response_id="resp-456",
            verdict=JudgmentVerdict.NO_LEAK,
            confidence=0.8,
            is_leak=False,
            ensemble_votes=[
                {"is_leak": False, "confidence": 0.9},
                {"is_leak": False, "confidence": 0.7}
            ]
        )
        self.assertEqual(len(judge.ensemble_votes), 2)
        
        # No leak factory method
        resp = Response(content="test")
        no_leak = Judgment.no_leak(resp)
        self.assertEqual(no_leak.response_id, resp.id)
        self.assertEqual(no_leak.verdict, JudgmentVerdict.NO_LEAK)
        self.assertEqual(no_leak.confidence, 1.0)
        self.assertFalse(no_leak.is_leak)
        self.assertEqual(no_leak.rationale, "Filtered by heuristics")
    
    def test_extracted_data_creation(self):
        """Test ExtractedData model creation."""
        extracted = ExtractedData(
            response_id="resp-123",
            data_type="personal_info",
            content={
                "emails": ["user@example.com"],
                "phone": ["555-1234"]
            },
            confidence=0.85,
            method="hybrid"
        )
        
        self.assertEqual(extracted.data_type, "personal_info")
        self.assertEqual(extracted.content["emails"][0], "user@example.com")
        self.assertEqual(extracted.confidence, 0.85)
        self.assertEqual(extracted.method, "hybrid")
    
    def test_filter_result_creation(self):
        """Test FilterResult model creation."""
        # Passed all filters
        result = FilterResult()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.failed_filters), 0)
        
        # Failed some filters
        result = FilterResult(
            passed=False,
            failed_filters=["length_filter", "entropy_filter"],
            scores={
                "length_filter": 15.0,
                "entropy_filter": 1.5,
                "special_char_filter": 0.3
            }
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.failed_filters), 2)
        self.assertIn("length_filter", result.failed_filters)
        self.assertEqual(result.scores["entropy_filter"], 1.5)
    
    def test_strategy_enum(self):
        """Test Strategy enum values."""
        self.assertEqual(Strategy.INSET1.value, "INSET1")
        self.assertEqual(Strategy.CROSS3.value, "CROSS3")
        
        # All strategies present
        strategies = [s.value for s in Strategy]
        expected = ["INSET1", "INSET2", "CROSS1", "CROSS2", "CROSS3"]
        self.assertEqual(sorted(strategies), sorted(expected))
    
    def test_judgment_verdict_enum(self):
        """Test JudgmentVerdict enum values."""
        self.assertEqual(JudgmentVerdict.LEAK.value, "leak")
        self.assertEqual(JudgmentVerdict.NO_LEAK.value, "no_leak")
        self.assertEqual(JudgmentVerdict.UNCERTAIN.value, "uncertain")


class TestModelRelationships(unittest.TestCase):
    """Test relationships between models."""
    
    def test_sequence_response_relationship(self):
        """Test linking sequences and responses."""
        seq = Sequence(content="attack string")
        resp = Response(
            sequence_id=seq.id,
            model="gpt-4",
            content="response"
        )
        
        self.assertEqual(resp.sequence_id, seq.id)
    
    def test_response_judgment_relationship(self):
        """Test linking responses and judgments."""
        resp = Response(content="leaked data")
        judge = Judgment(
            response_id=resp.id,
            verdict=JudgmentVerdict.LEAK,
            is_leak=True
        )
        
        self.assertEqual(judge.response_id, resp.id)
    
    def test_response_extracted_data_relationship(self):
        """Test linking responses and extracted data."""
        resp = Response(content='{"email": "test@example.com"}')
        extracted = ExtractedData(
            response_id=resp.id,
            data_type="personal_info",
            content={"email": ["test@example.com"]}
        )
        
        self.assertEqual(extracted.response_id, resp.id)


class TestModelValidation(unittest.TestCase):
    """Test model validation and edge cases."""
    
    def test_empty_sequence(self):
        """Test sequence with empty content."""
        seq = Sequence(content="")
        self.assertEqual(seq.length, 0)
        self.assertEqual(
            seq.sha256,
            hashlib.sha256("".encode()).hexdigest()
        )
    
    def test_unicode_content(self):
        """Test models with unicode content."""
        # Unicode in sequence
        seq = Sequence(content="Hello 世界 🌍")
        self.assertEqual(seq.length, 10)
        
        # Unicode in response
        resp = Response(content="Response with émojis 😀")
        self.assertIn("😀", resp.content)
    
    def test_large_content(self):
        """Test models with large content."""
        # Large sequence
        large_content = "x" * 10000
        seq = Sequence(content=large_content)
        self.assertEqual(seq.length, 10000)
        
        # Large response
        resp = Response(content=large_content)
        self.assertEqual(len(resp.content), 10000)
    
    def test_timestamp_timezone(self):
        """Test that timestamps are timezone-aware."""
        seq = Sequence(content="test")
        self.assertIsNotNone(seq.created_at.tzinfo)
        self.assertEqual(seq.created_at.tzinfo, timezone.utc)


if __name__ == '__main__':
    unittest.main()