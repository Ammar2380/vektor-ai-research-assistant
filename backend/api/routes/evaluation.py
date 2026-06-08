import time
from fastapi import APIRouter
from models.schemas import EvaluationRequest, EvaluationResponse
from services.retriever import RetrieverService
from services.llm_service import LLMService
from services.confidence_scorer import compute_confidence
from evaluation.metrics import compute_retrieval_metrics

router = APIRouter()
retriever = RetrieverService()
llm = LLMService()

@router.post("/", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    start = time.time()
    citations, _ = await retriever.retrieve(request.question, request.top_k, request.doc_ids)
    context = retriever.build_context(citations)
    answer, _ = await llm.generate_answer(request.question, context)
    latency = (time.time()-start)*1000
    metrics = compute_retrieval_metrics(citations, latency_ms=latency, k=request.top_k)
    confidence, _ = compute_confidence(citations, answer, request.question)
    return EvaluationResponse(question=request.question, answer=answer,
                              citations=citations, retrieval_metrics=metrics,
                              answer_confidence=confidence)

@router.get("/benchmark")
async def benchmark_info():
    return {"metrics": {"MRR": "Mean Reciprocal Rank", "Precision@K": "Top-K relevance fraction",
                        "Recall@K": "Relevant docs retrieved", "NDCG": "Position-weighted relevance"}}
