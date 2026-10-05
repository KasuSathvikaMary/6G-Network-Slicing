import pandas as pd
import numpy as np

class MetricsCollector:
    def __init__(self):
        self.data = []
        
    def record_step(self, step_data: dict) -> None:
        self.data.append(step_data)
        
    def get_episode_metrics(self) -> dict:
        if not self.data:
            return {}
        df = pd.DataFrame(self.data)
        return {
            'utilization_mean': df['utilization'].mean() if 'utilization' in df else 0.0,
            'violation_rate': np.mean([v.mean() for v in df['violations']]) if 'violations' in df else 0.0,
            'dissatisfaction_mean': df['dissatisfaction'].mean() if 'dissatisfaction' in df else 0.0,
            'isolation_violations_total': df['isolation_violations'].sum() if 'isolation_violations' in df else 0,
            'energy_mean': df['energy'].mean() if 'energy' in df else 0.0,
            'reward_total': df['reward'].sum() if 'reward' in df else 0.0,
            'reward_mean': df['reward'].mean() if 'reward' in df else 0.0
        }
        
    def reset(self) -> None:
        self.data = []
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)
