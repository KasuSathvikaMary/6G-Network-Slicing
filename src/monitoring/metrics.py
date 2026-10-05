import time
from typing import Dict, List, Any
from collections import defaultdict

class MetricsCollector:
    def __init__(self, retention_hours: int = 24):
        self.retention_seconds = retention_hours * 3600
        self.metrics = defaultdict(list)
        
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None):
        timestamp = time.time()
        self.metrics[metric_name].append({
            "timestamp": timestamp,
            "value": value,
            "tags": tags or {}
        })
        self._cleanup(metric_name)
        
    def _cleanup(self, metric_name: str):
        current_time = time.time()
        cutoff = current_time - self.retention_seconds
        self.metrics[metric_name] = [
            m for m in self.metrics[metric_name] 
            if m["timestamp"] >= cutoff
        ]
        
    def get_metrics(self, metric_name: str, tags_filter: Dict[str, str] = None) -> List[Dict]:
        if metric_name not in self.metrics:
            return []
            
        result = self.metrics[metric_name]
        if tags_filter:
            result = [
                m for m in result
                if all(m["tags"].get(k) == v for k, v in tags_filter.items())
            ]
        return result
        
    def get_time_series(self, metric_name: str, tags_filter: Dict[str, str] = None) -> Dict[str, List[float]]:
        data = self.get_metrics(metric_name, tags_filter)
        return {
            "timestamps": [m["timestamp"] for m in data],
            "values": [m["value"] for m in data]
        }
        
    def export_prometheus(self) -> str:
        lines = []
        for metric_name, data in self.metrics.items():
            if not data:
                continue
            
            lines.append(f"# HELP {metric_name} Metric {metric_name}")
            lines.append(f"# TYPE {metric_name} gauge")
            
            # Export latest value for each unique tag combination
            latest = {}
            for m in data:
                tag_str = ",".join(f'{k}="{v}"' for k, v in m["tags"].items())
                latest[tag_str] = m
                
            for tag_str, m in latest.items():
                tags_formatted = f"{{{tag_str}}}" if tag_str else ""
                lines.append(f"{metric_name}{tags_formatted} {m['value']} {int(m['timestamp']*1000)}")
                
        return "\n".join(lines)
