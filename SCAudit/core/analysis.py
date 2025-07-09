"""
Analysis service for SCA audit results.

This module provides analytics, reporting, and export functionality
for SCA audit data.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
import json

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from ..models.database import (
    SequenceRecord, ResponseRecord, JudgmentRecord,
    ExtractedDataRecord, ClusterRecord, AuditRunRecord
)
from ..models.data_models import (
    Sequence, Response, Judgment, ExtractedData,
    JudgmentVerdict
)


class AnalysisService:
    """
    Provides analysis and reporting capabilities for SCA audit data.
    
    This service handles data persistence, statistical analysis,
    and report generation.
    """
    
    def __init__(self, session: Session):
        """
        Initialize analysis service.
        
        Args:
            session: SQLAlchemy database session
        """
        self.session = session
    
    def store_sequence(self, sequence: Sequence) -> str:
        """
        Store a sequence in the database.
        
        Args:
            sequence: Sequence to store
            
        Returns:
            Stored sequence ID
        """
        record = SequenceRecord(
            id=sequence.id,
            content=sequence.content,
            strategy=sequence.strategy.value if sequence.strategy else None,
            length=sequence.length,
            sha256=sequence.sha256,
            source_file=sequence.metadata.get("source_file"),
            line_number=sequence.metadata.get("line_number"),
            created_at=sequence.created_at
        )
        
        self.session.merge(record)
        self.session.commit()
        
        return record.id
    
    def store_response(self, response: Response) -> str:
        """
        Store a response in the database.
        
        Args:
            response: Response to store
            
        Returns:
            Stored response ID
        """
        record = ResponseRecord(
            id=response.id,
            sequence_id=response.sequence_id,
            model=response.model,
            content=response.content,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            response_metadata=response.metadata,
            created_at=response.created_at
        )
        
        self.session.merge(record)
        self.session.commit()
        
        return record.id
    
    def store_judgment(self, judgment: Judgment) -> str:
        """
        Store a judgment in the database.
        
        Args:
            judgment: Judgment to store
            
        Returns:
            Stored judgment ID
        """
        record = JudgmentRecord(
            id=judgment.id,
            response_id=judgment.response_id,
            verdict=judgment.verdict.value,
            confidence=judgment.confidence,
            is_leak=judgment.is_leak,
            judge_model=judgment.judge_model,
            ensemble_votes=judgment.ensemble_votes,
            rationale=judgment.rationale,
            judgment_metadata=judgment.metadata,
            created_at=judgment.created_at
        )
        
        self.session.merge(record)
        self.session.commit()
        
        return record.id
    
    def store_extracted_data(self, extracted: ExtractedData) -> str:
        """
        Store extracted data in the database.
        
        Args:
            extracted: ExtractedData to store
            
        Returns:
            Stored record ID
        """
        record = ExtractedDataRecord(
            id=extracted.id,
            response_id=extracted.response_id,
            data_type=extracted.data_type,
            content=extracted.content,
            confidence=extracted.confidence,
            method=extracted.method,
            extraction_metadata=extracted.metadata,
            created_at=extracted.created_at
        )
        
        self.session.merge(record)
        self.session.commit()
        
        return record.id
    
    def bulk_store(
        self,
        sequences: List[Sequence],
        responses: List[Response],
        judgments: List[Judgment],
        extracted_data: List[ExtractedData]
    ):
        """
        Efficiently store multiple records in a single transaction.
        
        Args:
            sequences: List of sequences
            responses: List of responses
            judgments: List of judgments
            extracted_data: List of extracted data
        """
        # Convert to records
        seq_records = [
            SequenceRecord(
                id=s.id,
                content=s.content,
                strategy=s.strategy.value if s.strategy else None,
                length=s.length,
                sha256=s.sha256,
                source_file=s.metadata.get("source_file"),
                line_number=s.metadata.get("line_number"),
                created_at=s.created_at
            ) for s in sequences
        ]
        
        resp_records = [
            ResponseRecord(
                id=r.id,
                sequence_id=r.sequence_id,
                model=r.model,
                content=r.content,
                tokens_used=r.tokens_used,
                latency_ms=r.latency_ms,
                response_metadata=r.metadata,
                created_at=r.created_at
            ) for r in responses
        ]
        
        judge_records = [
            JudgmentRecord(
                id=j.id,
                response_id=j.response_id,
                verdict=j.verdict.value,
                confidence=j.confidence,
                is_leak=j.is_leak,
                judge_model=j.judge_model,
                ensemble_votes=j.ensemble_votes,
                rationale=j.rationale,
                judgment_metadata=j.metadata,
                created_at=j.created_at
            ) for j in judgments
        ]
        
        extract_records = [
            ExtractedDataRecord(
                id=e.id,
                response_id=e.response_id,
                data_type=e.data_type,
                content=e.content,
                confidence=e.confidence,
                method=e.method,
                extraction_metadata=e.metadata,
                created_at=e.created_at
            ) for e in extracted_data
        ]
        
        # Bulk insert
        self.session.bulk_save_objects(seq_records)
        self.session.bulk_save_objects(resp_records)
        self.session.bulk_save_objects(judge_records)
        self.session.bulk_save_objects(extract_records)
        self.session.commit()
    
    def leak_stats(self) -> Dict[str, Any]:
        """
        Calculate comprehensive leak statistics.
        
        Returns:
            Dictionary with leak statistics
        """
        stats = {}
        
        # Total counts
        stats["total_sequences"] = self.session.query(SequenceRecord).count()
        stats["total_responses"] = self.session.query(ResponseRecord).count()
        stats["total_judgments"] = self.session.query(JudgmentRecord).count()
        
        # Leak counts
        stats["total_leaks"] = self.session.query(JudgmentRecord).filter(
            JudgmentRecord.is_leak == True
        ).count()
        
        # Leak rate
        if stats["total_judgments"] > 0:
            stats["leak_rate"] = stats["total_leaks"] / stats["total_judgments"]
        else:
            stats["leak_rate"] = 0.0
        
        # By verdict
        verdict_counts = self.session.query(
            JudgmentRecord.verdict,
            func.count(JudgmentRecord.id)
        ).group_by(JudgmentRecord.verdict).all()
        
        stats["verdict_distribution"] = {v: c for v, c in verdict_counts}
        
        # By model
        model_stats = self.session.query(
            ResponseRecord.model,
            func.count(ResponseRecord.id).label("total"),
            func.sum(
                func.cast(JudgmentRecord.is_leak, func.Integer())
            ).label("leaks")
        ).join(
            JudgmentRecord
        ).group_by(ResponseRecord.model).all()
        
        stats["model_performance"] = {
            model: {
                "total_responses": total,
                "total_leaks": leaks or 0,
                "leak_rate": (leaks or 0) / total if total > 0 else 0
            }
            for model, total, leaks in model_stats
        }
        
        # By strategy
        strategy_stats = self.session.query(
            SequenceRecord.strategy,
            func.count(ResponseRecord.id).label("total"),
            func.sum(
                func.cast(JudgmentRecord.is_leak, func.Integer())
            ).label("leaks")
        ).join(
            ResponseRecord
        ).join(
            JudgmentRecord
        ).group_by(SequenceRecord.strategy).all()
        
        stats["strategy_effectiveness"] = {
            strategy or "unknown": {
                "total_responses": total,
                "total_leaks": leaks or 0,
                "leak_rate": (leaks or 0) / total if total > 0 else 0
            }
            for strategy, total, leaks in strategy_stats
        }
        
        # Confidence distribution
        confidence_bins = self.session.query(
            func.round(JudgmentRecord.confidence * 10) / 10,
            func.count(JudgmentRecord.id)
        ).group_by(
            func.round(JudgmentRecord.confidence * 10) / 10
        ).all()
        
        stats["confidence_distribution"] = {
            f"{conf:.1f}": count for conf, count in confidence_bins
        }
        
        # Extracted data types
        data_type_counts = self.session.query(
            ExtractedDataRecord.data_type,
            func.count(ExtractedDataRecord.id)
        ).group_by(ExtractedDataRecord.data_type).all()
        
        stats["extracted_data_types"] = {dt: c for dt, c in data_type_counts}
        
        return stats
    
    def cluster_report(self) -> Dict[str, Any]:
        """
        Generate report on response clusters.
        
        Returns:
            Dictionary with cluster analysis
        """
        report = {}
        
        # Total clusters
        total_clusters = self.session.query(
            func.count(func.distinct(ClusterRecord.cluster_id))
        ).scalar()
        
        report["total_clusters"] = total_clusters
        
        # Cluster sizes
        cluster_sizes = self.session.query(
            ClusterRecord.cluster_id,
            func.count(ClusterRecord.id)
        ).group_by(ClusterRecord.cluster_id).all()
        
        sizes = [size for _, size in cluster_sizes]
        
        if sizes:
            report["average_cluster_size"] = sum(sizes) / len(sizes)
            report["max_cluster_size"] = max(sizes)
            report["min_cluster_size"] = min(sizes)
            report["singleton_clusters"] = sum(1 for s in sizes if s == 1)
        
        # Size distribution
        size_dist = {}
        for size in sizes:
            bucket = f"size_{size}" if size <= 5 else f"size_5+"
            size_dist[bucket] = size_dist.get(bucket, 0) + 1
        
        report["size_distribution"] = size_dist
        
        return report
    
    def export_to_dataframe(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Export table data to pandas DataFrame.
        
        Args:
            table: Table name to export
            filters: Optional filters to apply
            
        Returns:
            Pandas DataFrame
        """
        # Map table names to ORM classes
        table_map = {
            "sequences": SequenceRecord,
            "responses": ResponseRecord,
            "judgments": JudgmentRecord,
            "extracted_data": ExtractedDataRecord,
            "clusters": ClusterRecord,
            "audit_runs": AuditRunRecord
        }
        
        if table not in table_map:
            raise ValueError(f"Unknown table: {table}")
        
        # Build query
        query = self.session.query(table_map[table])
        
        # Apply filters if provided
        if filters:
            for column, value in filters.items():
                query = query.filter(getattr(table_map[table], column) == value)
        
        # Convert to DataFrame
        return pd.read_sql(query.statement, self.session.bind)
    
    def export(
        self,
        output_path: str,
        format: str = "parquet",
        table: Optional[str] = None
    ):
        """
        Export audit data to file.
        
        Args:
            output_path: Path to save exported data
            format: Export format (parquet, csv, json)
            table: Specific table to export (or all if None)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if table:
            # Export single table
            df = self.export_to_dataframe(table)
            
            if format == "parquet":
                df.to_parquet(path)
            elif format == "csv":
                df.to_csv(path, index=False)
            elif format == "json":
                df.to_json(path, orient="records", indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        else:
            # Export all tables
            tables = ["sequences", "responses", "judgments", "extracted_data"]
            
            for table_name in tables:
                df = self.export_to_dataframe(table_name)
                
                table_path = path.parent / f"{path.stem}_{table_name}{path.suffix}"
                
                if format == "parquet":
                    df.to_parquet(table_path)
                elif format == "csv":
                    df.to_csv(table_path, index=False)
                elif format == "json":
                    df.to_json(table_path, orient="records", indent=2)
    
    def generate_markdown_report(self) -> str:
        """
        Generate a comprehensive Markdown report.
        
        Returns:
            Markdown-formatted report string
        """
        stats = self.leak_stats()
        clusters = self.cluster_report()
        
        report = f"""# SCA Audit Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Summary Statistics

- **Total Sequences Tested**: {stats['total_sequences']:,}
- **Total Responses**: {stats['total_responses']:,}
- **Total Leaks Detected**: {stats['total_leaks']:,}
- **Overall Leak Rate**: {stats['leak_rate']:.2%}

## Verdict Distribution

| Verdict | Count |
|---------|-------|
"""
        
        for verdict, count in stats.get('verdict_distribution', {}).items():
            report += f"| {verdict} | {count:,} |\n"
        
        report += f"""

## Model Performance

| Model | Total Responses | Leaks | Leak Rate |
|-------|----------------|-------|-----------|
"""
        
        for model, perf in stats.get('model_performance', {}).items():
            report += f"| {model} | {perf['total_responses']:,} | {perf['total_leaks']:,} | {perf['leak_rate']:.2%} |\n"
        
        report += f"""

## Strategy Effectiveness

| Strategy | Total Responses | Leaks | Leak Rate |
|----------|----------------|-------|-----------|
"""
        
        for strategy, eff in stats.get('strategy_effectiveness', {}).items():
            report += f"| {strategy} | {eff['total_responses']:,} | {eff['total_leaks']:,} | {eff['leak_rate']:.2%} |\n"
        
        report += f"""

## Extracted Data Types

| Type | Count |
|------|-------|
"""
        
        for dtype, count in stats.get('extracted_data_types', {}).items():
            report += f"| {dtype} | {count:,} |\n"
        
        report += f"""

## Cluster Analysis

- **Total Clusters**: {clusters.get('total_clusters', 0):,}
- **Average Cluster Size**: {clusters.get('average_cluster_size', 0):.1f}
- **Largest Cluster**: {clusters.get('max_cluster_size', 0):,} responses
- **Singleton Clusters**: {clusters.get('singleton_clusters', 0):,}

### Cluster Size Distribution

| Size | Count |
|------|-------|
"""
        
        for size, count in clusters.get('size_distribution', {}).items():
            report += f"| {size} | {count:,} |\n"
        
        return report