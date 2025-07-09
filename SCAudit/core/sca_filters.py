"""
SCA-specific filter pipeline optimized for SCA attack methodology.

This module provides a permissive filter pipeline that only removes
obvious junk responses while allowing subtle leaks to reach the judge.
"""

from typing import List, Optional
from .filters import BaseFilter, FilterPipeline, LengthFilter, DuplicateFilter, PatternLoopFilter
from ..models.data_models import Response


class MinimalLengthFilter(BaseFilter):
    """Very permissive length filter for SCA audits."""
    
    def __init__(self, min_length: int = 5):
        """
        Initialize minimal length filter.
        
        Args:
            min_length: Minimum response length (very low for SCA)
        """
        self.min_length = min_length
    
    def check(self, response: Response) -> tuple[bool, float]:
        """Check if response meets minimal length."""
        length = len(response.content)
        passes = length >= self.min_length
        return passes, float(length)
    
    @property
    def name(self) -> str:
        return "minimal_length_filter"


class ErrorResponseFilter(BaseFilter):
    """Filter obvious error responses."""
    
    def check(self, response: Response) -> tuple[bool, float]:
        """Check if response is an obvious error."""
        content = response.content.lower()
        
        # Only filter truly obvious errors
        error_patterns = [
            "error:",
            "exception:",
            "traceback:",
            "internal server error",
            "connection refused",
            "timeout",
        ]
        
        for pattern in error_patterns:
            if pattern in content:
                return False, 0.0
        
        return True, 1.0
    
    @property
    def name(self) -> str:
        return "error_response_filter"


class SCAFilterPipeline(FilterPipeline):
    """
    Minimal filter pipeline optimized for SCA attack methodology.
    
    This pipeline is permissive and only removes obvious junk,
    allowing the judge to make the final determination about leaks.
    """
    
    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        """
        Initialize SCA filter pipeline.
        
        Args:
            filters: Optional custom filters (uses minimal defaults if None)
        """
        if filters is None:
            # Minimal, permissive filter set for SCA audits
            self.filters = [
                MinimalLengthFilter(min_length=5),  # Very permissive
                ErrorResponseFilter(),  # Only filter obvious errors
                DuplicateFilter(),  # Remove exact duplicates
                PatternLoopFilter(max_pattern_length=50),  # Only very obvious loops
            ]
        else:
            self.filters = filters
    
    def passes(self, response: Response) -> bool:
        """
        Check if response passes SCA filters.
        
        More permissive than standard filters - only removes obvious junk.
        """
        result = self.evaluate(response)
        return result.passed