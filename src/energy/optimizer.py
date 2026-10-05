import numpy as np
from typing import List, Dict, Any, Tuple
from .model import EnergyModel

class EnergyOptimizer:
    """
    Multi-objective optimization: minimize energy while maintaining QoS.
    """
    def __init__(self, energy_model: EnergyModel, max_power_limit: float):
        self.energy_model = energy_model
        self.max_power_limit = max_power_limit
        
    def optimize(self, current_allocations: Dict[str, Dict[str, float]], qos_requirements: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Perform energy-aware scheduling and load consolidation.
        Returns optimized allocations.
        """
        optimized = {}
        total_power = 0.0
        
        # Simple consolidation heuristic: reduce resources for slices meeting QoS comfortably
        for slice_id, alloc in current_allocations.items():
            qos_target = qos_requirements.get(slice_id, 0.5)
            # Simulated current QoS
            current_qos = 0.8 # Placeholder for actual QoS metrics
            
            new_alloc = alloc.copy()
            
            if current_qos > qos_target * 1.2:
                # Can consolidate/reduce
                new_alloc['cpu'] = max(0.1, new_alloc['cpu'] * 0.9)
                new_alloc['mem'] = max(0.1, new_alloc['mem'] * 0.9)
                
            optimized[slice_id] = new_alloc
            total_power += self.energy_model.compute_energy(new_alloc['cpu'], new_alloc['mem'], new_alloc.get('net', 0.1))
            
        # Apply capping if needed
        if total_power > self.max_power_limit:
            optimized = self.apply_power_cap(optimized, total_power)
            
        return optimized

    def apply_power_cap(self, allocations: Dict[str, Dict[str, float]], current_power: float) -> Dict[str, Dict[str, float]]:
        """
        Throttle allocations to meet max power limit.
        """
        scaling_factor = self.max_power_limit / current_power
        
        throttled = {}
        for slice_id, alloc in allocations.items():
            throttled[slice_id] = {
                'cpu': alloc['cpu'] * scaling_factor,
                'mem': alloc['mem'] * scaling_factor,
                'net': alloc.get('net', 0.1) * scaling_factor
            }
        return throttled

    def get_pareto_front(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract Pareto front from candidate solutions (Energy vs QoS penalty).
        """
        pareto_front = []
        for i, cand1 in enumerate(candidates):
            dominated = False
            for j, cand2 in enumerate(candidates):
                if i == j:
                    continue
                # Assuming lower energy is better and lower QoS penalty is better
                if cand2['energy'] <= cand1['energy'] and cand2['qos_penalty'] <= cand1['qos_penalty'] and \
                   (cand2['energy'] < cand1['energy'] or cand2['qos_penalty'] < cand1['qos_penalty']):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(cand1)
        return pareto_front
