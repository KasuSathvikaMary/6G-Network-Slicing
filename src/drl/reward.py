import numpy as np

class RewardCalculator:
    def __init__(self, weights=None):
        if weights is None:
            self.weights = {
                'latency': 1.0,
                'throughput': 1.0,
                'reliability': 1.0,
                'energy': 0.5,
                'sla_violation': 2.0,
                'resource_efficiency': 0.5,
                'isolation': 1.0
            }
        else:
            self.weights = weights
            
    def compute_reward(self, metrics: dict, requirements: dict, alpha: float = 0.5, beta: float = 2.0, gamma: float = 0.5) -> float:
        """
        Compute the composite reward based on the specified components.
        
        metrics: dict containing 'actual_latency', 'actual_throughput', 'actual_reliability', 'energy_consumed', 'max_energy', 'violations', 'utilized', 'allocated', 'interference'
        requirements: dict containing 'max_latency', 'required_throughput', 'required_reliability'
        """
        # 1. QoS satisfaction component
        # Latency satisfaction
        actual_latency = metrics.get('actual_latency', 0)
        max_latency = requirements.get('max_latency', 1)
        latency_sat = max(0.0, 1.0 - (actual_latency / max_latency)) if max_latency > 0 else 0.0
        
        # Throughput satisfaction
        actual_throughput = metrics.get('actual_throughput', 0)
        req_throughput = requirements.get('required_throughput', 1)
        throughput_sat = min(actual_throughput / req_throughput, 1.0) if req_throughput > 0 else 0.0
        
        # Reliability satisfaction
        actual_reliability = metrics.get('actual_reliability', 0)
        req_reliability = requirements.get('required_reliability', 1)
        reliability_sat = (actual_reliability / req_reliability) if req_reliability > 0 else 0.0
        
        qos_reward = (self.weights['latency'] * latency_sat + 
                      self.weights['throughput'] * throughput_sat + 
                      self.weights['reliability'] * reliability_sat) / 3.0
                      
        # 2. Energy efficiency component
        energy_consumed = metrics.get('energy_consumed', 0)
        max_energy = metrics.get('max_energy', 1)
        energy_penalty = -alpha * (energy_consumed / max_energy) if max_energy > 0 else 0.0
        
        # 3. SLA violation penalty
        violations = metrics.get('violations', 0)
        sla_penalty = -beta * violations
        
        # 4. Resource efficiency bonus
        utilized = metrics.get('utilized', 0)
        allocated = metrics.get('allocated', 1)
        resource_bonus = gamma * (utilized / allocated) if allocated > 0 else 0.0
        
        # 5. Slice isolation penalty
        interference = metrics.get('interference', 0)
        isolation_penalty = -self.weights['isolation'] * interference
        
        # Composite reward
        total_reward = (qos_reward + 
                        self.weights['energy'] * energy_penalty + 
                        self.weights['sla_violation'] * sla_penalty + 
                        self.weights['resource_efficiency'] * resource_bonus + 
                        isolation_penalty)
                        
        # Reward clipping and shaping
        clipped_reward = np.clip(total_reward, -10.0, 10.0)
        
        return float(clipped_reward)
