from typing import Dict

class ModelValidator:
    def __init__(self):
        pass

    def validate(self, model_id: str) -> bool:
        return True

    def run_ab_test(self, model_a: str, model_b: str) -> Dict[str, float]:
        return {model_a: 0.8, model_b: 0.9}

    def compare_models(self, baseline: str, candidate: str) -> str:
        return candidate

    def is_improvement(self, metrics_baseline: dict, metrics_candidate: dict) -> bool:
        return metrics_candidate.get('reward', 0) > metrics_baseline.get('reward', 0)
