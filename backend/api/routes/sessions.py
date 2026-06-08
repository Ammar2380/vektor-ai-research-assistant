from datetime import datetime
from fastapi import APIRouter, HTTPException
from models.schemas import SessionCreate, SessionResponse, SessionHistory, ChatMessage
from services.session_manager import SessionManager

router = APIRouter()
sessions = SessionManager()

@router.post("/", response_model=SessionResponse, status_code=201)
async def create_session(body: SessionCreate):
    s = sessions.create_session(name=body.name, doc_ids=body.doc_ids or [])
    return SessionResponse(**s)

@router.get("/")
async def list_sessions():
    return [SessionResponse(**s).dict() for s in sessions.list_sessions()]

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    s = sessions.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found.")
    return SessionResponse(**s)

@router.get("/{session_id}/history", response_model=SessionHistory)
async def get_history(session_id: str, limit: int = 20):
    s = sessions.get_session(session_id)
    if not s: raise HTTPException(404, "Session not found.")
    history = sessions.get_history(session_id, limit)
    messages = [ChatMessage(role=m["role"], content=m["content"],
                            timestamp=datetime.fromisoformat(m["timestamp"])) for m in history]
    return SessionHistory(session_id=session_id, messages=messages, total_messages=len(messages))

@router.delete("/{session_id}")
async def delete_session(session_id: str):
    if not sessions.delete_session(session_id):
        raise HTTPException(404, "Session not found.")
    return {"deleted": True, "session_id": session_id}
