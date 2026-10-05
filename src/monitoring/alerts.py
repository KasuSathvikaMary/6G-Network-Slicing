import time
from dataclasses import dataclass
from typing import List, Dict, Callable
from enum import Enum

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    metric: str
    condition: Callable[[float], bool]
    severity: Severity
    message: str
    cooldown_seconds: int = 300

@dataclass
class Alert:
    id: str
    rule_metric: str
    severity: Severity
    message: str
    timestamp: float
    value: float
    acknowledged: bool = False

class AlertManager:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        self.rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.history: List[Alert] = []
        self.last_triggered: Dict[str, float] = {}
        
    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)
        
    def check_alerts(self):
        current_time = time.time()
        for rule in self.rules:
            # Check cooldown
            rule_id = f"{rule.metric}_{rule.severity.name}"
            if rule_id in self.last_triggered:
                if current_time - self.last_triggered[rule_id] < rule.cooldown_seconds:
                    continue
                    
            data = self.metrics.get_metrics(rule.metric)
            if not data:
                continue
                
            latest_val = data[-1]["value"]
            if rule.condition(latest_val):
                alert_id = f"alert_{int(current_time)}_{rule.metric}"
                alert = Alert(
                    id=alert_id,
                    rule_metric=rule.metric,
                    severity=rule.severity,
                    message=f"{rule.message} (Value: {latest_val})",
                    timestamp=current_time,
                    value=latest_val
                )
                self.active_alerts[alert_id] = alert
                self.history.append(alert)
                self.last_triggered[rule_id] = current_time
                
    def get_active_alerts(self) -> List[Alert]:
        return list(self.active_alerts.values())
        
    def acknowledge_alert(self, alert_id: str):
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            del self.active_alerts[alert_id]
