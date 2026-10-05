from typing import List

class ModelMonitor:
    def __init__(self):
        self.predictions: List[float] = []

    def record_prediction(self, pred: float):
        self.predictions.append(pred)

    def check_drift(self) -> bool:
        if len(self.predictions) < 100:
            return False
        recent = sum(self.predictions[-10:]) / 10
        old = sum(self.predictions[:10]) / 10
        return abs(recent - old) > 0.5

    def get_performance_metrics(self) -> dict:
        return {"accuracy": 0.95}

    def should_retrain(self) -> bool:
        return self.check_drift()
