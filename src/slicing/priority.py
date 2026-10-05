from enum import IntEnum
from typing import Dict

class PriorityLevel(IntEnum):
    BEST_EFFORT = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

class PriorityManager:
    def __init__(self):
        self.priorities: Dict[str, PriorityLevel] = {}

    def get_priority(self, slice_id: str) -> PriorityLevel:
        return self.priorities.get(slice_id, PriorityLevel.BEST_EFFORT)

    def adjust_priority(self, slice_id: str, new_level: PriorityLevel):
        self.priorities[slice_id] = new_level

    def preempt(self, high_slice: str, low_slice: str) -> bool:
        p1 = self.get_priority(high_slice)
        p2 = self.get_priority(low_slice)
        return p1 > p2

    def shed_load(self) -> str:
        if not self.priorities:
            return ""
        lowest = min(self.priorities.keys(), key=lambda k: self.priorities[k])
        return lowest
