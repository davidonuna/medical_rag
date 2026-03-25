# app/rag/reranker.py

"""
Cross-encoder reranker for improving RAG retrieval quality.
Uses BAAI/bge-reranker-base model for relevance scoring.
"""

from sentence_transformers import CrossEncoder
from typing import List, Tuple


class Reranker:
    """
    Cross-encoder reranker for medical RAG.
    
    Cross-encoders provide more accurate relevance scoring than
    bi-encoders by jointly encoding query-document pairs.
    
    Model: BAAI/bge-reranker-base
    
    Example:
        reranker = Reranker()
        ranked = reranker.rerank(
            query="What medications?",
            documents=["doc1", "doc2", "doc3"],
            top_k=2
        )
        # Returns: [("relevant_doc", 0.95), ("doc2", 0.82)]
    """

    def __init__(self):
        """
        Initialize the cross-encoder reranker.
        Loads BAAI/bge-reranker-base model on construction.
        """
        self.model = CrossEncoder(
            "BAAI/bge-reranker-base",
            trust_remote_code=True
        )

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents by relevance to query.
        
        Args:
            query: The search query string
            documents: List of document strings to re-rank
            top_k: Number of top results to return (default: 3)
            
        Returns:
            List of (document, score) tuples sorted by relevance descending
        """
        # Create query-document pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]
        
        # Get relevance scores for all pairs
        scores = self.model.predict(pairs)

        # Sort by score descending
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:top_k]
