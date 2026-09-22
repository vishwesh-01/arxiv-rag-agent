import json
import uuid
from datetime import datetime
from pathlib import Path

SESSION_FILE = Path("sessions.json")

def _load():
    if not SESSION_FILE.exists():
        return {}
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def _save(data):
    SESSION_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def create(paper: dict | None = None) -> str:
    sessions = _load()
    sid = uuid.uuid4().hex[:8]
    now = datetime.now().isoformat()
    sessions[sid] = {
        "id": sid,
        "title": "New session",
        "paper": paper,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    _save(sessions)
    return sid

def list_sessions() -> list[dict]:
    return list(_load().values())

def get(sid: str) -> dict | None:
    return _load().get(sid)

def set_paper(sid: str, paper: dict):
    data = _load()
    data[sid]["paper"] = paper
    data[sid]["updated_at"] = datetime.now().isoformat()
    _save(data)

def add_message(sid: str, role: str, content: str, sources=None):
    data = _load()
    session = data[sid]
    session["messages"].append({
        "role": role,
        "content": content,
        "sources": sources or [],
        "timestamp": datetime.now().isoformat(),
    })
    session["updated_at"] = datetime.now().isoformat()
    if role == "user" and session["title"] == "New session":
        session["title"] = content[:60]
    _save(data)
