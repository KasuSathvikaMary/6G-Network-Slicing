class ModelRegistry:
    def __init__(self, base_path: str): pass
    def register(self, model, version: str, metadata: dict): pass
    def get_latest(self) -> tuple: return "", {}
    def get_model_path(self, version: str) -> str: return ""
    @property
    def versions(self) -> list: return []
