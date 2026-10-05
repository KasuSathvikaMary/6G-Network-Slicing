from dataclasses import dataclass
from typing import Dict, List, Optional
import time

@dataclass
class SLADefinition:
    latency_ms: float
    throughput_mbps: float
    reliability: float
    availability: float

@dataclass
class SLAViolation:
    slice_id: str
    metric: str
    severity: str
    timestamp: float

class SLAMonitor:
    def __init__(self):
        self.slas: Dict[str, SLADefinition] = {}
        self.violations: List[SLAViolation] = []

    def check_compliance(self, slice_id: str, metrics: Dict[str, float]) -> bool:
        if slice_id not in self.slas:
            return True
        sla = self.slas[slice_id]
        if metrics.get('latency', 0) > sla.latency_ms:
            self.handle_violation(slice_id, 'latency', 'major')
            return False
        return True

    def handle_violation(self, slice_id: str, metric: str, severity: str):
        violation = SLAViolation(slice_id, metric, severity, time.time())
        self.violations.append(violation)

    def get_compliance_report(self) -> Dict:
        return {"violations": len(self.violations), "history": self.violations}

    def predict_violation(self, slice_id: str, metrics_trend: List[float]) -> bool:
        if not metrics_trend: return False
        return sum(metrics_trend[-3:]) / 3 > self.slas.get(slice_id, SLADefinition(float('inf'), 0, 0, 0)).latency_ms
