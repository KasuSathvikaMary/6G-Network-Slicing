class ScenarioRunner:
    def __init__(self, simulator):
        self.simulator = simulator
        self.metrics = {}

    def setup(self, scenario_name: str):
        pass

    def run(self):
        self.simulator.run(100)

    def validate_results(self) -> bool:
        return True

    def get_metrics(self) -> dict:
        return self.metrics
