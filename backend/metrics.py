"""
Retrieval evaluation metrics.
Implements MRR, Precision@K, Recall@K, and NDCG.
"""

import math
from typing import List, Optional
from models.schemas import Citation, RetrievalMetrics


def reciprocal_rank(citations: List[Citation], relevant_threshold: float = 0.5) -> float:
    """Mean Reciprocal Rank — rewards first relevant result."""
    for rank, c in enumerate(citations, 1):
        if c.relevance_score >= relevant_threshold:
            return 1.0 / rank
    return 0.0


def precision_at_k(citations: List[Citation], k: int, threshold: float = 0.5) -> float:
    """Fraction of top-k results that are relevant."""
    top_k = citations[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for c in top_k if c.relevance_score >= threshold)
    return relevant / len(top_k)


def recall_at_k(
    citations: List[Citation],
    total_relevant: int,
    k: int,
    threshold: float = 0.5,
) -> float:
    """Fraction of relevant documents retrieved in top-k."""
    if total_relevant == 0:
        return 0.0
    top_k = citations[:k]
    retrieved_relevant = sum(1 for c in top_k if c.relevance_score >= threshold)
    return retrieved_relevant / total_relevant


def ndcg_at_k(citations: List[Citation], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    top_k = citations[:k]
    if not top_k:
        return 0.0

    def dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

    actual_scores = [c.relevance_score for c in top_k]
    ideal_scores = sorted(actual_scores, reverse=True)

    actual_dcg = dcg(actual_scores)
    ideal_dcg = dcg(ideal_scores)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_retrieval_metrics(
    citations: List[Citation],
    latency_ms: float,
    k: int = 5,
    total_relevant: Optional[int] = None,
) -> RetrievalMetrics:
    """Compute all retrieval metrics."""
    total_rel = total_relevant or max(1, len(citations))

    return RetrievalMetrics(
        mrr=round(reciprocal_rank(citations), 4),
        precision_at_k=round(precision_at_k(citations, k), 4),
        recall_at_k=round(recall_at_k(citations, total_rel, k), 4),
        ndcg=round(ndcg_at_k(citations, k), 4),
        avg_relevance_score=round(
            sum(c.relevance_score for c in citations) / len(citations)
            if citations else 0.0, 4
        ),
        retrieved_count=len(citations),
        latency_ms=round(latency_ms, 2),
    )
