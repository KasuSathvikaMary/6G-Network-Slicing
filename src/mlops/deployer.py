class ModelDeployer:
    def __init__(self):
        self.current_model = None
        self.deployments = []

    def deploy(self, model_id: str):
        self.current_model = model_id
        self.deployments.append(model_id)

    def rollback(self):
        if len(self.deployments) > 1:
            self.deployments.pop()
            self.current_model = self.deployments[-1]

    def canary_deploy(self, model_id: str, traffic_pct: float):
        pass

    def get_deployment_status(self) -> str:
        return f"Deployed: {self.current_model}"
