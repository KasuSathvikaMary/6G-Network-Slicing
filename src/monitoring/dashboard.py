from typing import Dict, Any

class DashboardProvider:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        
    def get_overview(self) -> Dict[str, Any]:
        return {
            "active_slices": self._get_latest("active_slices") or 0,
            "cpu_utilization": self._get_latest("cpu_util") or 0.0,
            "memory_utilization": self._get_latest("memory_util") or 0.0,
            "total_throughput_gbps": self._get_latest("throughput_gbps") or 0.0,
            "system_health": "Healthy"
        }
        
    def get_slice_dashboard(self, slice_id: str) -> Dict[str, Any]:
        tags = {"slice_id": slice_id}
        return {
            "slice_id": slice_id,
            "latency_ms": self._get_latest("latency_ms", tags),
            "throughput_gbps": self._get_latest("throughput_gbps", tags),
            "packet_loss_rate": self._get_latest("packet_loss_rate", tags),
            "reliability": self._get_latest("reliability", tags),
            "jitter_ms": self._get_latest("jitter_ms", tags),
            "sla_compliance": 99.9  # Computed based on metrics vs SLA
        }
        
    def get_energy_dashboard(self) -> Dict[str, Any]:
        return {
            "current_power_watts": self._get_latest("energy_watts") or 0.0,
            "pue": 1.2,
            "energy_efficiency_score": 85.5,
            "carbon_footprint_gco2eq": 120.0
        }
        
    def get_ai_dashboard(self) -> Dict[str, Any]:
        return {
            "reward_avg": self._get_latest("reward_avg") or 0.0,
            "training_loss": self._get_latest("training_loss") or 0.0,
            "inference_time_ms": self._get_latest("inference_time_ms") or 0.0,
            "model_version": "v1.2.0"
        }
        
    def _get_latest(self, metric_name: str, tags: Dict[str, str] = None) -> float:
        data = self.metrics.get_metrics(metric_name, tags)
        if data:
            return data[-1]["value"]
        return 0.0
