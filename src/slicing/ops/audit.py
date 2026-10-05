class AuditLog:
    def __init__(self, consensus_delay_ms: float = 0.0): pass
    def append(self, step, obs, action, reward) -> dict: pass
    def verify(self) -> tuple: return True, ""
    def get_overhead_ms(self) -> list: return []
def verify_audit_log(entries: list) -> tuple: return True, ""
