import numpy as np

class TrafficGenerator:
    def __init__(self, slice_configs: list, episode_length: int = 288, seed: int = 42):
        self.slice_configs = slice_configs
        self.n_slices = len(slice_configs)
        self.episode_length = episode_length
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def reset(self, seed: int = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        
    def step(self, t: int) -> tuple:
        demand_prb = np.zeros(self.n_slices)
        demand_cu = np.zeros(self.n_slices)
        
        for i, sc in enumerate(self.slice_configs):
            phase = sc.get('daily_phase', 0.0)
            daily_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * t / self.episode_length + phase)
            
            burst_prob = sc.get('burst_prob', 0.0)
            burst_magnitude = sc.get('burst_magnitude', 0.0)
            burst = self.rng.binomial(1, burst_prob) * burst_magnitude
            
            noise_std = sc.get('noise_std', 0.0)
            
            base_prb = sc.get('base_demand_prb', 0.0)
            noise_prb = self.rng.normal(0, noise_std * base_prb)
            d_prb = base_prb * daily_pattern * (1 + burst) + noise_prb
            demand_prb[i] = max(0.0, d_prb)
            
            base_cu = sc.get('base_demand_cu', 0.0)
            noise_cu = self.rng.normal(0, noise_std * base_cu)
            d_cu = base_cu * daily_pattern * (1 + burst) + noise_cu
            demand_cu[i] = max(0.0, d_cu)
            
        return demand_prb, demand_cu
