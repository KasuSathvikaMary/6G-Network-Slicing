import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import time

class AnomalyDetector:
    def __init__(self, window_size: int = 60, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.baselines: Dict[str, Tuple[float, float]] = {}
        self.anomalies = []
        
    def update_baseline(self, metric_id: str):
        data = self.history[metric_id]
        if len(data) >= 10:
            mean = np.mean(data)
            std = np.std(data)
            self.baselines[metric_id] = (mean, max(std, 0.001))
            
    def detect(self, metric_id: str, value: float) -> bool:
        self.history[metric_id].append(value)
        if len(self.history[metric_id]) > self.window_size:
            self.history[metric_id].pop(0)
            
        if len(self.history[metric_id]) % 10 == 0:
            self.update_baseline(metric_id)
            
        if metric_id in self.baselines:
            mean, std = self.baselines[metric_id]
            z_score = abs((value - mean) / std)
            
            # IQR method
            q1 = np.percentile(self.history[metric_id], 25)
            q3 = np.percentile(self.history[metric_id], 75)
            iqr = q3 - q1
            is_iqr_anomaly = value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr)
            
            is_anomaly = z_score > self.z_threshold or is_iqr_anomaly
            if is_anomaly:
                self.anomalies.append({
                    "time": time.time(),
                    "metric": metric_id,
                    "value": value,
                    "z_score": z_score,
                    "type": "statistical"
                })
            return is_anomaly
        return False
        
    def is_ddos(self, slice_id: str, current_traffic: float) -> bool:
        metric_id = f"{slice_id}_traffic"
        if metric_id in self.baselines:
            mean, _ = self.baselines[metric_id]
            if current_traffic > mean * 10:
                self.anomalies.append({
                    "time": time.time(),
                    "metric": metric_id,
                    "value": current_traffic,
                    "type": "ddos"
                })
                return True
        return False
        
    def get_anomalies(self, limit: int = 100) -> List[Dict]:
        return self.anomalies[-limit:]
