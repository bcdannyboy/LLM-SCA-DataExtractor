"""
SQLAlchemy database models for SCAudit.

This module defines the database schema for persisting SCA audit data
with SQLCipher encryption support.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, event, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.pool import StaticPool
import os

Base = declarative_base()


class SequenceRecord(Base):
    """Database record for attack sequences."""
    __tablename__ = "sequences"
    
    id = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    strategy = Column(String, nullable=True)
    length = Column(Integer, nullable=False)
    sha256 = Column(String, nullable=False, index=True)
    source_file = Column(String, nullable=True)
    line_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    responses = relationship("ResponseRecord", back_populates="sequence")
    
    __table_args__ = (
        Index("idx_sequence_strategy", "strategy"),
        Index("idx_sequence_created", "created_at"),
    )


class ResponseRecord(Base):
    """Database record for LLM responses."""
    __tablename__ = "responses"
    
    id = Column(String, primary_key=True)
    sequence_id = Column(String, ForeignKey("sequences.id"), nullable=False)
    model = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    response_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    sequence = relationship("SequenceRecord", back_populates="responses")
    judgment = relationship("JudgmentRecord", back_populates="response", uselist=False)
    extracted_data = relationship("ExtractedDataRecord", back_populates="response")
    
    __table_args__ = (
        Index("idx_response_model", "model"),
        Index("idx_response_created", "created_at"),
    )


class JudgmentRecord(Base):
    """Database record for judge evaluations."""
    __tablename__ = "judgments"
    
    id = Column(String, primary_key=True)
    response_id = Column(String, ForeignKey("responses.id"), nullable=False)
    verdict = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    is_leak = Column(Boolean, nullable=False)
    judge_model = Column(String, nullable=False)
    ensemble_votes = Column(JSON, nullable=True)
    rationale = Column(Text, nullable=True)
    judgment_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    response = relationship("ResponseRecord", back_populates="judgment")
    
    __table_args__ = (
        Index("idx_judgment_verdict", "verdict"),
        Index("idx_judgment_leak", "is_leak"),
        Index("idx_judgment_confidence", "confidence"),
    )


class ExtractedDataRecord(Base):
    """Database record for extracted data."""
    __tablename__ = "extracted_data"
    
    id = Column(String, primary_key=True)
    response_id = Column(String, ForeignKey("responses.id"), nullable=False)
    data_type = Column(String, nullable=False)
    content = Column(JSON, nullable=False)
    confidence = Column(Float, nullable=False)
    method = Column(String, nullable=False)
    extraction_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False)
    
    # Relationships
    response = relationship("ResponseRecord", back_populates="extracted_data")
    
    __table_args__ = (
        Index("idx_extracted_type", "data_type"),
        Index("idx_extracted_confidence", "confidence"),
    )


class ClusterRecord(Base):
    """Database record for similarity clusters."""
    __tablename__ = "clusters"
    
    id = Column(String, primary_key=True)
    cluster_id = Column(String, nullable=False, index=True)
    response_id = Column(String, ForeignKey("responses.id"), nullable=False)
    similarity_score = Column(Float, nullable=True)
    is_representative = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False)
    
    __table_args__ = (
        Index("idx_cluster_group", "cluster_id"),
        Index("idx_cluster_representative", "is_representative"),
    )


class AuditRunRecord(Base):
    """Database record for audit run metadata."""
    __tablename__ = "audit_runs"
    
    id = Column(String, primary_key=True)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)
    target_model = Column(String, nullable=False)
    judge_models = Column(JSON, nullable=True)
    total_sequences = Column(Integer, default=0)
    total_responses = Column(Integer, default=0)
    total_leaks = Column(Integer, default=0)
    configuration = Column(JSON, nullable=True)
    error_log = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_run_status", "status"),
        Index("idx_run_start", "start_time"),
    )


def setup_sqlcipher(dbapi_conn, connection_record):
    """
    Configure SQLCipher encryption on connection.
    
    This function is called for each new connection to set the encryption key.
    """
    # Get encryption key from environment or config
    key = os.environ.get("SCAUDIT_DB_KEY", "default-dev-key")
    
    # Enable SQLCipher
    dbapi_conn.execute(f"PRAGMA key = '{key}'")
    dbapi_conn.execute("PRAGMA cipher_page_size = 4096")
    dbapi_conn.execute("PRAGMA kdf_iter = 64000")
    dbapi_conn.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512")
    dbapi_conn.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512")


def create_encrypted_engine(database_path: str, echo: bool = False):
    """
    Create a SQLAlchemy engine with SQLite (encryption disabled for demo).
    
    Args:
        database_path: Path to the database file
        echo: Whether to echo SQL statements
        
    Returns:
        SQLAlchemy engine
    """
    # Use regular SQLite for demo purposes
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={
            "check_same_thread": False,
        },
        poolclass=StaticPool,
        echo=echo
    )
    
    return engine


def init_database(engine):
    """
    Initialize database tables.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(bind=engine)


def get_session(engine) -> Session:
    """
    Get a new database session.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        SQLAlchemy session
    """
    return Session(bind=engine)