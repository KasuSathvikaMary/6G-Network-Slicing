import random

class TrafficGenerator:
    def __init__(self):
        self.pattern = "poisson"

    def set_pattern(self, pattern: str):
        self.pattern = pattern

    def generate_traffic(self, slice_type: str) -> float:
        if slice_type == "eMBB":
            return random.uniform(10, 100)
        elif slice_type == "uRLLC":
            return 5.0
        return random.expovariate(1.0)

    def get_next_arrival(self) -> float:
        return random.expovariate(2.0)
