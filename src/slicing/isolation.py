from typing import Dict
import logging

logger = logging.getLogger(__name__)

class IsolationManager:
    def __init__(self):
        self.hard_limits: Dict[str, float] = {}
        self.soft_limits: Dict[str, float] = {}
        self.current_usage: Dict[str, float] = {}

    def check_isolation(self, slice_id: str) -> bool:
        return self.current_usage.get(slice_id, 0.0) <= self.hard_limits.get(slice_id, float('inf'))

    def enforce_boundaries(self) -> None:
        for slice_id, usage in self.current_usage.items():
            limit = self.hard_limits.get(slice_id, float('inf'))
            if usage > limit:
                logger.warning(f"Slice {slice_id} exceeded hard limit. Throttling.")
                self.current_usage[slice_id] = limit

    def get_isolation_score(self) -> float:
        violations = sum(1 for s, u in self.current_usage.items() if u > self.soft_limits.get(s, float('inf')))
        return max(0.0, 1.0 - (violations * 0.1))

    def detect_interference(self) -> bool:
        return self.get_isolation_score() < 0.8
