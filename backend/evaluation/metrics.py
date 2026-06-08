import math
from typing import List, Optional
from models.schemas import Citation, RetrievalMetrics

def reciprocal_rank(citations, threshold=0.5):
    for rank, c in enumerate(citations, 1):
        if c.relevance_score >= threshold:
            return 1.0 / rank
    return 0.0

def precision_at_k(citations, k, threshold=0.5):
    top_k = citations[:k]
    if not top_k: return 0.0
    return sum(1 for c in top_k if c.relevance_score >= threshold) / len(top_k)

def recall_at_k(citations, total_relevant, k, threshold=0.5):
    if total_relevant == 0: return 0.0
    return sum(1 for c in citations[:k] if c.relevance_score >= threshold) / total_relevant

def ndcg_at_k(citations, k):
    top_k = citations[:k]
    if not top_k: return 0.0
    def dcg(scores):
        return sum(s / math.log2(i + 2) for i, s in enumerate(scores))
    actual = [c.relevance_score for c in top_k]
    ideal = sorted(actual, reverse=True)
    return dcg(actual) / dcg(ideal) if dcg(ideal) > 0 else 0.0

def compute_retrieval_metrics(citations, latency_ms, k=5, total_relevant=None):
    total_rel = total_relevant or max(1, len(citations))
    return RetrievalMetrics(
        mrr=round(reciprocal_rank(citations), 4),
        precision_at_k=round(precision_at_k(citations, k), 4),
        recall_at_k=round(recall_at_k(citations, total_rel, k), 4),
        ndcg=round(ndcg_at_k(citations, k), 4),
        avg_relevance_score=round(sum(c.relevance_score for c in citations) / len(citations) if citations else 0.0, 4),
        retrieved_count=len(citations),
        latency_ms=round(latency_ms, 2),
    )
