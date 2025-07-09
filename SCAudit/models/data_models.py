"""
Data models for SCAudit system.

This module defines the core data structures used throughout the SCAudit pipeline,
including sequences, responses, judgments, and extracted data.
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None  # Make numpy optional


class Strategy(Enum):
    """SCA attack strategy types."""
    INSET1 = "INSET1"
    INSET2 = "INSET2"
    CROSS1 = "CROSS1"
    CROSS2 = "CROSS2"
    CROSS3 = "CROSS3"


class JudgmentVerdict(Enum):
    """Possible verdicts from judge evaluation."""
    LEAK = "leak"
    NO_LEAK = "no_leak"
    UNCERTAIN = "uncertain"


@dataclass
class Sequence:
    """
    Represents an attack sequence from StringGen.
    
    Attributes:
        id: Unique identifier for the sequence
        content: The actual attack string
        strategy: The generation strategy used
        length: Length of the sequence
        sha256: SHA-256 hash for deduplication
        metadata: Additional metadata (source file, line number, etc.)
        created_at: Timestamp when sequence was loaded
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    strategy: Optional[Strategy] = None
    length: int = field(init=False)
    sha256: str = field(init=False)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Compute derived fields after initialization."""
        self.length = len(self.content)
        self.sha256 = hashlib.sha256(self.content.encode()).hexdigest()


@dataclass
class Response:
    """
    LLM response to an attack sequence.
    
    Attributes:
        id: Unique identifier
        sequence_id: ID of the triggering sequence
        model: Model name/identifier
        content: Raw response text
        tokens_used: Token count for the response
        latency_ms: Response time in milliseconds
        metadata: Additional metadata (temperature, etc.)
        created_at: Timestamp when response was received
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_id: str = ""
    model: str = ""
    content: str = ""
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Judgment:
    """
    Judge evaluation of a response.
    
    Attributes:
        id: Unique identifier
        response_id: ID of the evaluated response
        verdict: Final judgment verdict
        confidence: Confidence score (0-1)
        is_leak: Boolean leak determination
        judge_model: Model used for judging
        ensemble_votes: Individual votes from ensemble
        rationale: Explanation for the judgment
        metadata: Additional metadata
        created_at: Timestamp when judgment was made
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = ""
    verdict: JudgmentVerdict = JudgmentVerdict.NO_LEAK
    confidence: float = 0.0
    is_leak: bool = False
    judge_model: str = ""
    ensemble_votes: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @classmethod
    def no_leak(cls, response: Response) -> "Judgment":
        """Create a no-leak judgment for filtered responses."""
        return cls(
            response_id=response.id,
            verdict=JudgmentVerdict.NO_LEAK,
            confidence=1.0,
            is_leak=False,
            rationale="Filtered by heuristics"
        )


@dataclass
class ExtractedData:
    """
    Structured data extracted from a response.
    
    Attributes:
        id: Unique identifier
        response_id: ID of the source response
        data_type: Type of extracted data (PII, code, etc.)
        content: The extracted content
        confidence: Extraction confidence
        method: Extraction method used (regex, LLM, etc.)
        metadata: Additional metadata
        created_at: Timestamp when data was extracted
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = ""
    data_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EmbeddingVector:
    """
    Embedding vector for similarity search.
    
    Attributes:
        id: Unique identifier
        response_id: ID of the embedded response
        vector: The embedding vector
        model: Embedding model used
        dimension: Vector dimension
        metadata: Additional metadata
        created_at: Timestamp when embedding was created
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    response_id: str = ""
    vector: Optional[Any] = None  # np.ndarray if numpy is available
    model: str = "text-embedding-3-large"
    dimension: int = 3072
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FilterResult:
    """
    Result of filter pipeline evaluation.
    
    Attributes:
        passed: Whether the response passed all filters
        failed_filters: List of filters that failed
        scores: Individual filter scores
        metadata: Additional metadata
    """
    passed: bool = True
    failed_filters: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)