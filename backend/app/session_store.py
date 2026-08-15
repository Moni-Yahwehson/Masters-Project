"""In-memory conversation store.

Deliberately simple: this is a thesis prototype, so sessions live in process
memory with a TTL and are lost on restart. Swap the implementation for Redis or
a database if the deployment needs durability — the interface is small.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .config import get_settings
from .schemas import DiagnosisReport


@dataclass
class StoredMessage:
    id: str
    role: str  # "user" | "assistant"
    text: str = ""
    markdown: str = ""
    title: str = ""
    kind: str = "knowledge"
    image_count: int = 0
    report: Optional[DiagnosisReport] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    title: str = "New Chat"
    messages: list[StoredMessage] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._ttl = get_settings().session_ttl_seconds

    def _purge_expired(self) -> None:
        cutoff = time.time() - self._ttl
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.updated_at < cutoff
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        with self._lock:
            self._purge_expired()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.updated_at = time.time()
                return session
            new_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
            session = Session(id=new_id)
            self._sessions[new_id] = session
            return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            self._purge_expired()
            return self._sessions.get(session_id)

    def append(self, session_id: str, message: StoredMessage) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = Session(id=session_id)
                self._sessions[session_id] = session
            session.messages.append(message)
            session.updated_at = time.time()
            if session.title == "New Chat" and message.role == "user":
                text = message.text.strip() or "Image diagnosis"
                session.title = text[:40] + ("…" if len(text) > 40 else "")

    def history(self, session_id: str, limit: int = 20) -> list[StoredMessage]:
        session = self.get(session_id)
        if session is None:
            return []
        return session.messages[-limit:]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[Session]:
        with self._lock:
            self._purge_expired()
            return sorted(
                self._sessions.values(), key=lambda s: s.updated_at, reverse=True
            )


_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
