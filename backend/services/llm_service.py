from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from core.config import settings
from core.logging_config import setup_logging

logger = setup_logging(__name__)

# FIX: Moved the system prompt to a clean, minimal baseline. 
# The heavy-lifting guardrails are now supplied via the `context` parameter from chat.py.
SYSTEM_PROMPT = "You are an elite AI system. Process the provided directives and queries precisely."

class LLMService:
    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=settings.LLM_TEMPERATURE if hasattr(settings, 'LLM_TEMPERATURE') else 0.0, # Optimized to 0.0 for deterministic output
                max_tokens=1024,
            )
        return self._llm

    async def generate_answer(self, question, context, history=[]):
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Hydrate memory
        for turn in history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
                
        # FIX: Directly pass the secured context and question without redundant formatting
        user_msg = f"{context}\n\nUser Question: {question}"
        messages.append(HumanMessage(content=user_msg))
        
        try:
            llm = self._get_llm()
            response = llm.invoke(messages)
            # Estimate tokens safely
            token_count = len(response.content.split()) * 4 // 3
            return response.content, token_count
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error calling Groq API: {str(e)}\n\nRetrieved context:\n{context[:500]}", 0