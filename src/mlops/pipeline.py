from typing import List
import time

class RetrainingPipeline:
    def __init__(self):
        self.last_retrain = time.time()
        self.history = []

    def check_triggers(self, metrics: dict) -> bool:
        if metrics.get('degradation', 0) > 0.2:
            return True
        if time.time() - self.last_retrain > 86400:
            return True
        return False

    def start_retraining(self) -> str:
        self.last_retrain = time.time()
        return "training_job_123"

    def validate_model(self, job_id: str) -> bool:
        return True

    def deploy_model(self, job_id: str) -> bool:
        if self.validate_model(job_id):
            self.history.append(job_id)
            return True
        return False
