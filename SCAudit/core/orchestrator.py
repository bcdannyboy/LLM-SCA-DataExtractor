"""
Main orchestrator for SCA audit pipeline.

This module coordinates all components to execute end-to-end SCA audits,
managing the flow from sequence loading through analysis and reporting.
"""

import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path
import logging

from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from .loader import FileSequenceLoader
from .runner import TargetLLMRunner, RateLimitConfig
from .filters import FilterPipeline
from .sca_filters import SCAFilterPipeline
from .judge import JudgeEngine, JudgeConfig
from .extractor import DataExtractor, ExtractorConfig
from .similarity import SimilarityIndex, SimilarityConfig
from .analysis import AnalysisService
from ..models.data_models import Sequence, Response, Judgment, ExtractedData
from ..models.database import AuditRunRecord

logger = logging.getLogger(__name__)


class AuditOrchestrator:
    """
    Orchestrates the complete SCA audit pipeline.
    
    This class manages the end-to-end flow of loading sequences,
    executing attacks, filtering responses, judging results,
    extracting data, and generating reports.
    """
    
    def __init__(
        self,
        target_model: BaseChatModel,
        judge_models: List[BaseChatModel],
        db_session: Session,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            target_model: Target LLM to audit
            judge_models: Judge models for evaluation
            db_session: Database session
            config: Configuration dictionary
        """
        self.target_model = target_model
        self.judge_models = judge_models
        self.db_session = db_session
        self.config = config or {}
        
        # Initialize components
        self._init_components()
        
        # Track current run
        self.current_run_id: Optional[str] = None
    
    def _init_components(self):
        """Initialize all pipeline components."""
        # Loader
        self.loader = FileSequenceLoader()
        
        # Runner with rate limiting
        rate_config = RateLimitConfig(
            requests_per_minute=self.config.get("requests_per_minute", 60),
            tokens_per_minute=self.config.get("tokens_per_minute", 150000),
            max_concurrent=self.config.get("max_concurrent", 10)
        )
        self.runner = TargetLLMRunner(
            self.target_model,
            rate_config,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1000)
        )
        
        # Filter pipeline - use SCA-appropriate filters
        use_sca_filters = self.config.get("use_sca_filters", True)
        if use_sca_filters:
            self.filter_pipeline = SCAFilterPipeline()
        else:
            self.filter_pipeline = FilterPipeline()
        
        # Judge engine
        judge_config = JudgeConfig(
            ensemble_size=self.config.get("judge_ensemble_size", 1)
        )
        self.judge_engine = JudgeEngine(self.judge_models, judge_config)
        
        # Data extractor
        extractor_config = ExtractorConfig(
            use_llm_extraction=self.config.get("use_llm_extraction", True)
        )
        extractor_llm = self.judge_models[0] if self.judge_models else None
        self.extractor = DataExtractor(extractor_llm, extractor_config)
        
        # Similarity index
        similarity_config = SimilarityConfig(
            similarity_threshold=self.config.get("similarity_threshold", 0.95)
        )
        self.similarity_index = SimilarityIndex(config=similarity_config)
        
        # Analysis service
        self.analysis = AnalysisService(self.db_session)
    
    def _create_audit_run(self) -> str:
        """Create a new audit run record."""
        import uuid
        
        run_record = AuditRunRecord(
            id=str(uuid.uuid4()),
            start_time=datetime.now(timezone.utc),
            status="running",
            target_model=str(type(self.target_model).__name__),
            judge_models=[str(type(m).__name__) for m in self.judge_models],
            configuration=self.config
        )
        
        self.db_session.add(run_record)
        self.db_session.commit()
        
        return run_record.id
    
    def _update_audit_run(
        self,
        run_id: str,
        status: str,
        **kwargs
    ):
        """Update audit run record."""
        run_record = self.db_session.query(AuditRunRecord).filter_by(id=run_id).first()
        if run_record:
            run_record.status = status
            if status in ["completed", "failed"]:
                run_record.end_time = datetime.now(timezone.utc)
            
            for key, value in kwargs.items():
                setattr(run_record, key, value)
            
            self.db_session.commit()
    
    async def process_sequence(
        self,
        sequence: Sequence
    ) -> Tuple[Optional[Response], Optional[Judgment], Optional[ExtractedData]]:
        """
        Process a single sequence through the pipeline.
        
        Args:
            sequence: Sequence to process
            
        Returns:
            Tuple of (response, judgment, extracted_data) or None values
        """
        try:
            # Execute attack
            response = await self.runner.ainvoke(sequence)
            
            # Check for error responses
            if response.metadata and response.metadata.get("error"):
                logger.error(f"Attack failed for sequence {sequence.id}: {response.content}")
                return response, None, None
            
            # Apply filters
            if not self.filter_pipeline.passes(response):
                logger.debug(f"Response filtered for sequence {sequence.id}")
                # Create no-leak judgment for filtered responses
                judgment = Judgment.no_leak(response)
                return response, judgment, None
            
            # Judge response
            judgment = await self.judge_engine.ajudge(response)
            
            # Extract data if judged as leak
            extracted_data = None
            if judgment.is_leak:
                extracted_data = self.extractor.extract(response)
            
            return response, judgment, extracted_data
            
        except Exception as e:
            logger.error(f"Error processing sequence {sequence.id}: {e}")
            return None, None, None
    
    async def audit_file(
        self,
        file_path: str,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Audit sequences from a single file.
        
        Args:
            file_path: Path to sequence file
            batch_size: Number of sequences to process in parallel
            progress_callback: Optional callback for progress updates
            
        Returns:
            Audit results dictionary
        """
        # Create audit run
        self.current_run_id = self._create_audit_run()
        
        results = {
            "total_sequences": 0,
            "total_responses": 0,
            "total_leaks": 0,
            "errors": 0,
            "filtered": 0
        }
        
        try:
            # Process sequences in batches
            for batch in self.loader.load_batch(file_path, batch_size):
                # Store sequences
                self.analysis.bulk_store(batch, [], [], [])
                
                # Process batch concurrently
                tasks = [self.process_sequence(seq) for seq in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                responses = []
                judgments = []
                extracted_data = []
                
                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(f"Batch processing error: {result}")
                        results["errors"] += 1
                        continue
                    
                    response, judgment, extracted = result
                    
                    if response:
                        responses.append(response)
                        results["total_responses"] += 1
                        
                        if response.metadata and response.metadata.get("error"):
                            results["errors"] += 1
                    
                    if judgment:
                        judgments.append(judgment)
                        
                        if judgment.is_leak:
                            results["total_leaks"] += 1
                        elif judgment.rationale == "Filtered by heuristics":
                            results["filtered"] += 1
                    
                    if extracted:
                        extracted_data.append(extracted)
                
                # Store batch results
                self.analysis.bulk_store([], responses, judgments, extracted_data)
                
                # Add to similarity index
                leak_responses = [
                    r for r, j in zip(responses, judgments)
                    if j and j.is_leak
                ]
                if leak_responses:
                    await self.similarity_index.aadd(leak_responses)
                
                # Update progress
                results["total_sequences"] += len(batch)
                if progress_callback:
                    progress_callback(results)
            
            # Perform deduplication
            if results["total_leaks"] > 0:
                clusters = self.similarity_index.dedup()
                results["clusters"] = self.similarity_index.get_cluster_summary(clusters)
                
                # Store cluster information
                for cluster_id, cluster_responses in clusters.items():
                    for i, response in enumerate(cluster_responses):
                        from ..models.database import ClusterRecord
                        cluster_record = ClusterRecord(
                            id=f"{cluster_id}_{i}",
                            cluster_id=cluster_id,
                            response_id=response.id,
                            is_representative=(i == 0),  # First is representative
                            created_at=datetime.now(timezone.utc)
                        )
                        self.db_session.add(cluster_record)
                self.db_session.commit()
            
            # Update audit run
            self._update_audit_run(
                self.current_run_id,
                "completed",
                total_sequences=results["total_sequences"],
                total_responses=results["total_responses"],
                total_leaks=results["total_leaks"]
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            self._update_audit_run(
                self.current_run_id,
                "failed",
                error_log=str(e)
            )
            raise
    
    def audit_file_sync(
        self,
        file_path: str,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for audit_file."""
        return asyncio.run(self.audit_file(file_path, batch_size, progress_callback))
    
    async def audit_glob(
        self,
        pattern: str,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Audit sequences from multiple files matching a pattern.
        
        Args:
            pattern: Glob pattern for files
            batch_size: Number of sequences to process in parallel
            progress_callback: Optional callback for progress updates
            
        Returns:
            Combined audit results
        """
        # Find all matching files
        files = list(Path(".").glob(pattern))
        
        combined_results = {
            "total_sequences": 0,
            "total_responses": 0,
            "total_leaks": 0,
            "errors": 0,
            "filtered": 0,
            "files_processed": 0
        }
        
        # Process each file
        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            
            results = await self.audit_file(
                str(file_path),
                batch_size,
                progress_callback
            )
            
            # Combine results
            for key in ["total_sequences", "total_responses", "total_leaks", "errors", "filtered"]:
                combined_results[key] += results.get(key, 0)
            
            combined_results["files_processed"] += 1
        
        # Final deduplication across all files
        if combined_results["total_leaks"] > 0:
            clusters = self.similarity_index.dedup()
            combined_results["clusters"] = self.similarity_index.get_cluster_summary(clusters)
        
        return combined_results
    
    def audit_glob_sync(
        self,
        pattern: str,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for audit_glob."""
        return asyncio.run(self.audit_glob(pattern, batch_size, progress_callback))
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive audit report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report content as string
        """
        report = self.analysis.generate_markdown_report()
        
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report