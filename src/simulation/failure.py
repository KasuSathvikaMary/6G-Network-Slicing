from typing import List

class FailureInjector:
    def __init__(self):
        self.active_failures = []

    def inject_failure(self, target: str, failure_type: str):
        self.active_failures.append({"target": target, "type": failure_type})

    def schedule_failure(self, target: str, failure_type: str, time: float):
        pass

    def recover(self, target: str):
        self.active_failures = [f for f in self.active_failures if f["target"] != target]

    def get_active_failures(self) -> List[dict]:
        return self.active_failures
