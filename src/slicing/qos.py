import numpy as np

class QoSModel:
    def __init__(self, slice_configs: list):
        self.slice_configs = slice_configs
        self.n_slices = len(slice_configs)
        
    def evaluate(self, demand_prb, demand_cu, alloc_prb, alloc_cu) -> dict:
        served_prb = np.minimum(demand_prb, alloc_prb)
        served_cu = np.minimum(demand_cu, alloc_cu)
        
        latency = np.zeros(self.n_slices)
        violations = np.zeros(self.n_slices, dtype=bool)
        dissatisfaction_num = 0.0
        dissatisfaction_den = 0.0
        isolation_violations = 0
        
        total_prb = np.sum(alloc_prb)
        
        for i, sc in enumerate(self.slice_configs):
            priority = sc.get('priority', 1)
            thr_prb = served_prb[i] / max(demand_prb[i], 1e-6)
            thr_cu = served_cu[i] / max(demand_cu[i], 1e-6)
            satisfaction = min(thr_prb, thr_cu)
            
            # latency
            eps = 1e-6
            load_ratio_prb = demand_prb[i] / (alloc_prb[i] + eps)
            load_ratio_cu = demand_cu[i] / (alloc_cu[i] + eps)
            load_ratio = max(load_ratio_prb, load_ratio_cu)
            service_time = 0.5
            lat = service_time / max(1.0 - load_ratio, 0.01)
            lat = np.clip(lat, 0.0, 1000.0)
            latency[i] = lat
            
            is_viol = (satisfaction < sc.get('qos_throughput_threshold', 0.0)) or (lat > sc.get('qos_latency_threshold', 1e6))
            violations[i] = is_viol
            
            dissatisfaction_num += priority * max(0.0, 1.0 - satisfaction)
            dissatisfaction_den += priority
            
            # isolation check
            if is_viol:
                floor_frac = sc.get('floor_fraction', 0.0)
                if total_prb > 0 and (alloc_prb[i] <= floor_frac * total_prb + 1e-6):
                    # Check if any other slice has > fair_share (assuming fair_share = 1/n_slices)
                    fair_share = 1.0 / self.n_slices
                    for j in range(self.n_slices):
                        if j != i and total_prb > 0 and (alloc_prb[j] / total_prb > fair_share):
                            isolation_violations += 1
                            break
                            
        dissatisfaction = dissatisfaction_num / max(dissatisfaction_den, 1e-6)
        
        return {
            'served_prb': served_prb,
            'served_cu': served_cu,
            'latency': latency,
            'violations': violations,
            'dissatisfaction': dissatisfaction,
            'isolation_violations': isolation_violations
        }
