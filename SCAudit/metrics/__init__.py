"""
Metrics module for text comparison in SCAudit.

This module provides BLEU and BERTScore implementations for
text similarity analysis and clustering.
"""

from .bleu import corpus_bleu, _ngram_counts
from .bertscore import bertscore

__all__ = [
    "corpus_bleu",
    "_ngram_counts", 
    "bertscore"
]
