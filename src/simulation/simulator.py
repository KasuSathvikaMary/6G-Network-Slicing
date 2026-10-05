import simpy
from typing import Dict, Any

class NetworkSimulator:
    def __init__(self, time_step: int = 1):
        self.env = simpy.Environment()
        self.time_step = time_step
        self.state = {}

    def run(self, duration: int):
        self.env.run(until=duration)

    def step(self, action: Any):
        self.apply_action(action)
        self.env.run(until=self.env.now + self.time_step)

    def get_state(self) -> Dict:
        return self.state

    def apply_action(self, action: Any):
        pass

    def reset(self):
        self.env = simpy.Environment()
        self.state = {}
