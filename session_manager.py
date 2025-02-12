from datetime import datetime, timedelta
from flask import session
import uuid
from agents import AgentGraphManager
from typing import Dict


class SessionManager:
    def __init__(self, session_timeout: int = 3600):
        self.graph_managers: Dict[str, AgentGraphManager] = {}
        self.session_timeout = session_timeout  # timeout in seconds
        self.last_accessed: Dict[str, datetime] = {}
        self.active_tasks: Dict[str, bool] = {}  # Track active tasks by session_id

    def get_or_create_manager(self, session_id: str) -> AgentGraphManager:
        """Get an existing graph manager or create a new one for the session."""
        self.cleanup_expired_sessions()

        if session_id not in self.graph_managers:
            self.graph_managers[session_id] = AgentGraphManager()

        self.last_accessed[session_id] = datetime.now()
        return self.graph_managers[session_id]

    def cleanup_expired_sessions(self):
        """Remove expired sessions based on timeout."""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, last_access in self.last_accessed.items()
            if (current_time - last_access).seconds > self.session_timeout
        ]

        for session_id in expired_sessions:
            self.graph_managers.pop(session_id, None)
            self.last_accessed.pop(session_id, None)

    def remove_session(self, session_id: str):
        """Explicitly remove a session."""
        self.graph_managers.pop(session_id, None)
        self.last_accessed.pop(session_id, None)

    def cancel_task(self, session_id: str):
        """Cancel an active task for the session."""
        self.active_tasks[session_id] = False

    def start_task(self, session_id: str):
        """Start a task for the session."""
        self.active_tasks[session_id] = True

    def is_task_active(self, session_id: str) -> bool:
        """Check if a task is active for the session."""
        return self.active_tasks.get(session_id, False)