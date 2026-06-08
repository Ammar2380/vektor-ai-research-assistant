from typing import List
from models.schemas import Citation

def compute_confidence(citations, answer, query):
    if not citations:
        return 0.0, "No Sources"
        
    # FIX: Recognize deterministic guardrail activations
    guardrail_phrases = [
        "solution steps are not indexed",
        "couldn't find relevant information",
        "explicit solution steps are not indexed"
    ]
    
    if any(phrase in answer.lower() for phrase in guardrail_phrases):
        return 0.0, "Context Missing (Guardrail)"

    avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
    source_score = min(len(citations) / 3, 1.0)
    words = len(answer.split())
    
    if words < 20:
        answer_score = 0.5
    elif words < 50:
        answer_score = 0.7
    else:
        answer_score = 1.0
        
    unique_docs = len(set(c.doc_id for c in citations))
    diversity_score = min(unique_docs / 2, 1.0)
    
    score = round(min(max(
        0.40 * avg_relevance + 0.20 * source_score +
        0.20 * answer_score + 0.20 * diversity_score, 0.0), 1.0), 3)
        
    if score >= 0.75: label = "High Confidence"
    elif score >= 0.50: label = "Medium Confidence"
    elif score >= 0.25: label = "Low Confidence"
    else: label = "Very Low Confidence"
    
    return score, label