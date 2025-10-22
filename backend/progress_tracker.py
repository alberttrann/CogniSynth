# backend/progress_tracker.py
import asyncio
from typing import Dict, Optional, List
from datetime import datetime
import json

class ProgressTracker:
    """
    Track analysis progress for real-time updates via Server-Sent Events.
    Thread-safe for concurrent document processing.
    """
    
    def __init__(self):
        self._tasks: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def start_task(self, task_id: str, total_steps: int = 5):
        """Initialize a new task with progress tracking."""
        async with self._lock:
            self._tasks[task_id] = {
                "status": "processing",
                "current_step": 0,
                "total_steps": total_steps,
                "progress": 0,
                "message": "Initializing analysis...",
                "started_at": datetime.utcnow().isoformat(),
                "completed_at": None,
                "error": None
            }
    
    async def update_progress(self, task_id: str, step: int, message: str):
        """Update task progress with current step and message."""
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task["current_step"] = step
                task["progress"] = int((step / task["total_steps"]) * 100)
                task["message"] = message
    
    async def complete_task(self, task_id: str, success: bool = True, error: str = None):
        """Mark task as completed or failed."""
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].update({
                    "status": "completed" if success else "failed",
                    "progress": 100 if success else self._tasks[task_id]["progress"],
                    "message": "Analysis complete!" if success else f"Failed: {error}",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error": error
                })
    
    def get_progress(self, task_id: str) -> Optional[Dict]:
        """Get current progress for a task (synchronous for SSE)."""
        return self._tasks.get(task_id)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Remove completed tasks older than specified hours."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for task_id, task in self._tasks.items():
            if task["completed_at"]:
                completed_time = datetime.fromisoformat(task["completed_at"])
                if completed_time < cutoff:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        return len(to_remove)
    
    def get_active_tasks(self) -> List[Dict]:
        """Get all currently processing tasks."""
        return [
            {"task_id": tid, **task}
            for tid, task in self._tasks.items()
            if task["status"] == "processing"
        ]
    
    def get_task_summary(self) -> Dict:
        """Get summary statistics of all tasks."""
        total = len(self._tasks)
        processing = sum(1 for t in self._tasks.values() if t["status"] == "processing")
        completed = sum(1 for t in self._tasks.values() if t["status"] == "completed")
        failed = sum(1 for t in self._tasks.values() if t["status"] == "failed")
        
        return {
            "total": total,
            "processing": processing,
            "completed": completed,
            "failed": failed
        }

# Global singleton instance
progress_tracker = ProgressTracker()