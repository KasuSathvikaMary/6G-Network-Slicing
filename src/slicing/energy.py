class PowerModel:
    def __init__(self, p_idle=100.0, alpha=200.0, beta=100.0):
        self.p_idle = p_idle
        self.alpha = alpha
        self.beta = beta
        
    def compute_power(self, prb_used, prb_total, cu_used, cu_total) -> float:
        return self.p_idle + self.alpha * (prb_used / prb_total) + self.beta * (cu_used / cu_total)
        
    def normalized_energy(self, prb_used, prb_total, cu_used, cu_total) -> float:
        p_max = self.p_idle + self.alpha + self.beta
        return self.compute_power(prb_used, prb_total, cu_used, cu_total) / p_max
