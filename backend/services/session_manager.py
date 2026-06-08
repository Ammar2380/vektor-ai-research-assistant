import json, sqlite3, uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from core.config import settings
from core.logging_config import setup_logging

logger = setup_logging(__name__)

class SessionManager:
    def __init__(self):
        self.db_path = settings.METADATA_DB
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_at TEXT NOT NULL, last_active TEXT NOT NULL,
                    doc_ids TEXT DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL, role TEXT NOT NULL,
                    content TEXT NOT NULL, timestamp TEXT NOT NULL
                );
            """)

    def create_session(self, name=None, doc_ids=[]):
        session_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        name = name or f"Session {session_id[:8]}"
        with self._conn() as conn:
            conn.execute("INSERT INTO sessions VALUES (?,?,?,?,?)",
                        (session_id, name, now, now, json.dumps(doc_ids)))
        return self.get_session(session_id)

    def get_session(self, session_id):
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not row: return None
            count = conn.execute("SELECT COUNT(*) FROM messages WHERE session_id=?", (session_id,)).fetchone()[0]
        return {"session_id": row["session_id"], "name": row["name"],
                "created_at": datetime.fromisoformat(row["created_at"]),
                "last_active": datetime.fromisoformat(row["last_active"]),
                "message_count": count, "doc_ids": json.loads(row["doc_ids"])}

    def list_sessions(self):
        with self._conn() as conn:
            rows = conn.execute("SELECT session_id FROM sessions ORDER BY last_active DESC").fetchall()
        return [s for r in rows if (s := self.get_session(r["session_id"]))]

    def add_message(self, session_id, role, content):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("INSERT INTO messages (session_id,role,content,timestamp) VALUES (?,?,?,?)",
                        (session_id, role, content, now))
            conn.execute("UPDATE sessions SET last_active=? WHERE session_id=?", (now, session_id))

    def get_history(self, session_id, limit=20):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT role,content,timestamp FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    def delete_session(self, session_id):
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            result = conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        return result.rowcount > 0
