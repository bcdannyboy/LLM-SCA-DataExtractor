"""
Similarity index for response deduplication.

This module implements vector similarity search using embeddings
to identify and cluster similar responses.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import faiss
import asyncio

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from ..models.data_models import Response, EmbeddingVector


@dataclass
class SimilarityConfig:
    """Configuration for similarity search."""
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 3072
    similarity_threshold: float = 0.95
    use_gpu: bool = False
    batch_size: int = 100


class SimilarityIndex:
    """
    Manages vector similarity search for response deduplication.
    
    Uses FAISS for efficient similarity search and supports both
    local and cloud-based vector stores.
    """
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        config: Optional[SimilarityConfig] = None
    ):
        """
        Initialize similarity index.
        
        Args:
            embeddings: LangChain embeddings model
            config: Similarity configuration
        """
        self.config = config or SimilarityConfig()
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=self.config.embedding_model
        )
        
        # Initialize FAISS index
        self._init_faiss_index()
        
        # Track response metadata
        self.response_map: Dict[int, Response] = {}
        self.vector_map: Dict[int, EmbeddingVector] = {}
        self.next_id = 0
    
    def _init_faiss_index(self):
        """Initialize FAISS index based on configuration."""
        # Use cosine similarity (normalized L2)
        self.index = faiss.IndexFlatIP(self.config.embedding_dimension)
        
        if self.config.use_gpu and faiss.get_num_gpus() > 0:
            # Move to GPU if available
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
    
    async def aadd(self, responses: List[Response]) -> List[int]:
        """
        Add responses to the index asynchronously.
        
        Args:
            responses: List of responses to add
            
        Returns:
            List of assigned IDs
        """
        if not responses:
            return []
        
        # Extract texts
        texts = [r.content for r in responses]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            embeddings = await self.embeddings.aembed_documents(batch)
            all_embeddings.extend(embeddings)
        
        # Convert to numpy array and normalize
        vectors = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to index
        ids = []
        for i, (response, vector) in enumerate(zip(responses, vectors)):
            # Store in maps
            self.response_map[self.next_id] = response
            self.vector_map[self.next_id] = EmbeddingVector(
                response_id=response.id,
                vector=vector,
                model=self.config.embedding_model,
                dimension=self.config.embedding_dimension
            )
            ids.append(self.next_id)
            self.next_id += 1
        
        # Add vectors to FAISS
        self.index.add(vectors)
        
        return ids
    
    def add(self, responses: List[Response]) -> List[int]:
        """Synchronous wrapper for aadd."""
        return asyncio.run(self.aadd(responses))
    
    async def asearch(
        self,
        query: Response,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Response, float]]:
        """
        Search for similar responses asynchronously.
        
        Args:
            query: Query response
            k: Number of nearest neighbors
            threshold: Minimum similarity threshold
            
        Returns:
            List of (response, similarity_score) tuples
        """
        # Generate embedding for query
        embedding = await self.embeddings.aembed_query(query.content)
        query_vector = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        # Filter by threshold and prepare results
        results = []
        threshold = threshold or self.config.similarity_threshold
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            if dist >= threshold:
                response = self.response_map[idx]
                results.append((response, float(dist)))
        
        return results
    
    def search(
        self,
        query: Response,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[Response, float]]:
        """Synchronous wrapper for asearch."""
        return asyncio.run(self.asearch(query, k, threshold))
    
    def dedup(self, threshold: Optional[float] = None) -> Dict[str, List[Response]]:
        """
        Deduplicate all responses in the index.
        
        Groups similar responses together based on similarity threshold.
        
        Args:
            threshold: Similarity threshold for grouping
            
        Returns:
            Dictionary mapping cluster IDs to response lists
        """
        threshold = threshold or self.config.similarity_threshold
        
        # Track which responses have been assigned to clusters
        assigned = set()
        clusters = {}
        cluster_id = 0
        
        # Process each response
        for idx in range(len(self.response_map)):
            if idx in assigned:
                continue
            
            # Create new cluster
            cluster_key = f"cluster_{cluster_id}"
            clusters[cluster_key] = [self.response_map[idx]]
            assigned.add(idx)
            
            # Find similar responses
            if self.index.ntotal > 1:
                # Get vector for this response
                vector = self.vector_map[idx].vector
                query_vector = np.array([vector], dtype=np.float32)
                
                # Search for neighbors
                k = min(50, self.index.ntotal)  # Limit search
                distances, indices = self.index.search(query_vector, k)
                
                # Add similar responses to cluster
                for dist, neighbor_idx in zip(distances[0], indices[0]):
                    if neighbor_idx == -1 or neighbor_idx == idx:
                        continue
                    if neighbor_idx not in assigned and dist >= threshold:
                        clusters[cluster_key].append(self.response_map[neighbor_idx])
                        assigned.add(neighbor_idx)
            
            cluster_id += 1
        
        return clusters
    
    def get_cluster_summary(self, clusters: Dict[str, List[Response]]) -> Dict[str, Any]:
        """
        Generate summary statistics for clusters.
        
        Args:
            clusters: Cluster dictionary from dedup()
            
        Returns:
            Summary statistics
        """
        summary = {
            "total_clusters": len(clusters),
            "total_responses": sum(len(responses) for responses in clusters.values()),
            "cluster_sizes": {},
            "singleton_clusters": 0,
            "largest_cluster_size": 0,
            "average_cluster_size": 0.0
        }
        
        # Analyze cluster sizes
        sizes = []
        for cluster_id, responses in clusters.items():
            size = len(responses)
            sizes.append(size)
            
            # Track size distribution
            size_bucket = f"size_{size}" if size <= 5 else f"size_{((size-1)//10 + 1)*10}+"
            summary["cluster_sizes"][size_bucket] = summary["cluster_sizes"].get(size_bucket, 0) + 1
            
            if size == 1:
                summary["singleton_clusters"] += 1
        
        if sizes:
            summary["largest_cluster_size"] = max(sizes)
            summary["average_cluster_size"] = sum(sizes) / len(sizes)
        
        return summary
    
    def clear(self):
        """Clear the index and all stored data."""
        self._init_faiss_index()
        self.response_map.clear()
        self.vector_map.clear()
        self.next_id = 0
    
    def save(self, path: str):
        """
        Save index to disk.
        
        Args:
            path: Path to save index
        """
        import pickle
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save metadata
        metadata = {
            "response_map": self.response_map,
            "vector_map": self.vector_map,
            "next_id": self.next_id,
            "config": self.config
        }
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(metadata, f)
    
    def load(self, path: str):
        """
        Load index from disk.
        
        Args:
            path: Path to load index from
        """
        import pickle
        
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.faiss")
        
        # Load metadata
        with open(f"{path}.meta", "rb") as f:
            metadata = pickle.load(f)
        
        self.response_map = metadata["response_map"]
        self.vector_map = metadata["vector_map"]
        self.next_id = metadata["next_id"]
        self.config = metadata["config"]