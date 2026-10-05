import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EnergyProfile:
    """Hardware specifications for energy modeling."""
    p_base_cpu: float
    p_max_cpu: float
    p_base_mem: float
    p_max_mem: float
    p_base_net: float
    p_max_net: float
    cooling_multiplier: float
    alpha: float = 1.0 # Exponent for non-linear power scaling

class EnergyModel:
    """
    Energy consumption model computing energy per slice and per component.
    """
    def __init__(self, profile: EnergyProfile):
        self.profile = profile
        self.ema_energy = 0.0
        self.ema_alpha = 0.2
        
    def compute_energy(self, cpu_util: float, mem_util: float, net_util: float) -> float:
        """
        Compute total energy consumption: E = P_base + P_dynamic * utilization^alpha
        Utilizations should be between 0.0 and 1.0
        """
        cpu_util = max(0.0, min(1.0, cpu_util))
        mem_util = max(0.0, min(1.0, mem_util))
        net_util = max(0.0, min(1.0, net_util))
        
        p_cpu = self.profile.p_base_cpu + (self.profile.p_max_cpu - self.profile.p_base_cpu) * (cpu_util ** self.profile.alpha)
        p_mem = self.profile.p_base_mem + (self.profile.p_max_mem - self.profile.p_base_mem) * (mem_util ** self.profile.alpha)
        p_net = self.profile.p_base_net + (self.profile.p_max_net - self.profile.p_base_net) * (net_util ** self.profile.alpha)
        
        total_power = (p_cpu + p_mem + p_net) * self.profile.cooling_multiplier
        
        # Update EMA
        if self.ema_energy == 0.0:
            self.ema_energy = total_power
        else:
            self.ema_energy = self.ema_alpha * total_power + (1 - self.ema_alpha) * self.ema_energy
            
        return total_power

    def predict_energy(self, horizon: int = 1) -> float:
        """
        Predict energy using exponential moving average.
        """
        # Simplistic EMA-based prediction
        return self.ema_energy

    def get_energy_breakdown(self, cpu_util: float, mem_util: float, net_util: float) -> Dict[str, float]:
        """
        Get power consumption breakdown per component.
        """
        p_cpu = self.profile.p_base_cpu + (self.profile.p_max_cpu - self.profile.p_base_cpu) * (cpu_util ** self.profile.alpha)
        p_mem = self.profile.p_base_mem + (self.profile.p_max_mem - self.profile.p_base_mem) * (mem_util ** self.profile.alpha)
        p_net = self.profile.p_base_net + (self.profile.p_max_net - self.profile.p_base_net) * (net_util ** self.profile.alpha)
        
        cooling = (p_cpu + p_mem + p_net) * (self.profile.cooling_multiplier - 1.0)
        
        return {
            "cpu": p_cpu,
            "memory": p_mem,
            "network": p_net,
            "cooling": cooling,
            "total": p_cpu + p_mem + p_net + cooling
        }
