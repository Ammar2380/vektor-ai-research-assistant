"""
LLM service — wraps Ollama / OpenAI-compatible endpoint via LangChain.
Falls back to a simple template response if LLM is unreachable.
"""

import time
from typing import List, Optional
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from core.config import settings
from core.logging_config import setup_logging

logger = setup_logging(__name__)

SYSTEM_PROMPT = """You are an expert AI Research Assistant. Your role is to answer questions 
based ONLY on the provided context from research documents. 

Rules:
1. Answer ONLY from the provided context — never hallucinate.
2. If the context doesn't contain the answer, say: "I couldn't find relevant information in the uploaded documents."
3. Always be precise, concise, and cite specific sources when possible.
4. If multiple sources support the answer, synthesize them coherently.
5. Use bullet points or numbered lists when appropriate for clarity.
"""


class LLMService:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.LLM_API_BASE,
                temperature=settings.LLM_TEMPERATURE,
                num_predict=settings.LLM_MAX_TOKENS,
            )
        return self._llm

    async def generate_answer(
        self,
        question: str,
        context: str,
        history: List[dict] = [],
    ) -> tuple[str, int]:
        """
        Generate an answer using RAG context.
        Returns (answer_text, estimated_tokens).
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Add conversation history (last N turns)
        for turn in history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        # Current question with context
        user_message = f"""Context from research documents:
{context}

---

Question: {question}

Please answer based on the context above. Cite sources where applicable."""

        messages.append(HumanMessage(content=user_message))

        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            answer = response.content
            tokens = len(answer.split()) * 4 // 3  # rough estimate
            return answer, tokens

        except Exception as e:
            logger.warning(f"LLM unavailable ({e}), using fallback")
            return self._fallback_answer(question, context), 0

    def _fallback_answer(self, question: str, context: str) -> str:
        """Simple template-based fallback when LLM is unreachable."""
        if not context.strip():
            return "No relevant documents found. Please upload PDFs first."
        excerpt = context[:500].strip()
        return (
            f"Based on the uploaded documents, here is relevant information "
            f"related to your question:\n\n{excerpt}\n\n"
            f"*(Note: LLM service is currently unavailable. "
            f"Showing raw retrieved context.)*"
        )
