import time, uuid
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.retriever import RetrieverService
from services.llm_service import LLMService
from services.session_manager import SessionManager
from services.confidence_scorer import compute_confidence
from core.config import settings

router = APIRouter()
retriever = RetrieverService()
llm = LLMService()
sessions = SessionManager()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()
    
    # ── Session Resolution ───────────────────────────────────────────────────
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.get_session(session_id)
    if not session:
        session = sessions.create_session()
        session_id = session["session_id"]
    history = sessions.get_history(session_id)
    
    # ── Context Extraction Matrix ─────────────────────────────────────────────
    citations, avg_relevance = await retriever.retrieve(request.question, request.top_k, request.doc_ids)
    raw_context = retriever.build_context(citations)
    
    # ── Deterministic System Guardrails ───────────────────────────────────────
    # Wraps the retrieved documents in strict execution rules with separated logic paths
    secured_context = f"""
[CRITICAL ARCHITECTURAL DIRECTIVE]
You are Vektor, an elite AI Research Assistant. Answer the user question based strictly and exclusively on the provided context chunks.

1. GENERAL ABSENCE: If the context chunks do not contain the specific page, example number, or topic requested by the user, you MUST state exactly: "I couldn't find relevant information in the uploaded documents for that specific reference."

2. MATHEMATICAL CONTEXT GAPS: If the context contains a question, assignment prompt, or mathematical word problem but does NOT provide the step-by-step solution key, you MUST state: "The document contains the problem text, but the explicit solution steps are not indexed in the provided nodes."

3. GROUNDING: Never attempt to invent algebraic structures, set sizes, or Venn diagram intersections on your own. Do not extrapolate or calculate values that are not explicitly solved within the text nodes.

[KNOWLEDGE OVERLAY CONTEXT]
{raw_context}
[END OF CONTEXT]
"""

    # ── Inference Pipeline Execution ─────────────────────────────────────────
    answer, tokens = await llm.generate_answer(request.question, secured_context, history)
    
    # ── Telemetry & Evaluation Metrics ───────────────────────────────────────
    confidence, label = compute_confidence(citations, answer, request.question)
    
    # ── State Preservation ───────────────────────────────────────────────────
    sessions.add_message(session_id, "user", request.question)
    sessions.add_message(session_id, "assistant", answer)
    
    return ChatResponse(
        answer=answer, 
        session_id=session_id, 
        citations=citations,
        confidence_score=confidence, 
        confidence_label=label,
        retrieved_chunks=len(citations),
        processing_time_ms=(time.time() - start) * 1000,
        model_used=settings.GROQ_MODEL, 
        tokens_used=tokens
    )