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
        self.api_keys: Dict[str, str] = {}  # Store API keys by session

    def get_or_create_manager(self, session_id: str) -> AgentGraphManager:
        """Get an existing graph manager or create a new one for the session."""
        self.cleanup_expired_sessions()

        if session_id not in self.graph_managers:
            print(f"\n[DEBUG] Creating new graph manager for session {session_id}")
            self.graph_managers[session_id] = AgentGraphManager()
            # If we have an API key for this session, set it
            if api_key := self.get_api_key(session_id):
                print(f"\n[DEBUG] Found existing API key for new graph manager")
                self.graph_managers[session_id].update_api_key(api_key)

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
            self.api_keys.pop(session_id, None)  # Also clean up API keys

    def remove_session(self, session_id: str):
        """Explicitly remove a session."""
        self.graph_managers.pop(session_id, None)
        self.last_accessed.pop(session_id, None)
        self.api_keys.pop(session_id, None)  # Also remove API key

    def cancel_task(self, session_id: str):
        """Cancel an active task for the session."""
        self.active_tasks[session_id] = False
        # Make sure we record that cancellation happened
        print(f"\n[DEBUG] Task cancelled for session {session_id}")

    def reset_task_state(self, session_id: str):
        """Reset the task state for a session after errors or cancellation."""
        print(f"\n[DEBUG] Resetting task state for session {session_id}")
        self.active_tasks[session_id] = False
        
        # If the graph manager exists, ensure it's in a clean state
        if session_id in self.graph_managers:
            # Keep the manager but reset any task-specific state
            # This allows reusing the same session for new tasks
            pass
            
        # Update the last access time to prevent early cleanup
        self.last_accessed[session_id] = datetime.now()
        
        print(f"\n[DEBUG] Task state reset completed for session {session_id}")
        return True

    def start_task(self, session_id: str):
        """Start a task for the session."""
        # First make sure any previous task is properly cleaned up
        self.reset_task_state(session_id)
        self.active_tasks[session_id] = True
        print(f"\n[DEBUG] Task started for session {session_id}")

    def is_task_active(self, session_id: str) -> bool:
        """Check if a task is active for the session."""
        return self.active_tasks.get(session_id, False)

    def set_api_key(self, session_id: str, api_key: str) -> None:
        """Set the API key for a session."""
        print(f"\n[DEBUG] Setting API key in session manager for session {session_id}")
        self.api_keys[session_id] = api_key
        if session_id in self.graph_managers:
            print(f"\n[DEBUG] Updating existing graph manager with new API key")
            self.graph_managers[session_id].update_api_key(api_key)
        print(f"\n[DEBUG] API key set successfully in session manager")

    def get_api_key(self, session_id: str) -> str:
        """Get the API key for a session."""
        api_key = self.api_keys.get(session_id)
        print(f"\n[DEBUG] Retrieved API key for session {session_id}: {'Present' if api_key else 'Not found'}")
        return api_key

    def force_recreate_manager(self, session_id: str):
        """Force recreation of a graph manager for a session after severe errors.
        
        This is more drastic than reset_task_state as it completely rebuilds
        the graph manager, which can help recover from memory or state corruption.
        """
        print(f"\n[DEBUG] Force recreating graph manager for session {session_id}")
        self.active_tasks[session_id] = False
        
        # Preserve the API key if we have one
        api_key = self.get_api_key(session_id)
        
        # Remove existing graph manager
        if session_id in self.graph_managers:
            self.graph_managers.pop(session_id)
            
        # Create a fresh manager
        self.graph_managers[session_id] = AgentGraphManager()
        
        # Re-apply API key if we had one
        if api_key:
            self.graph_managers[session_id].update_api_key(api_key)
            
        # Update the last access time
        self.last_accessed[session_id] = datetime.now()
        
        print(f"\n[DEBUG] Graph manager forcefully recreated for session {session_id}")
        return True