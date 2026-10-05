import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .traffic import TrafficGenerator
from .energy import PowerModel
from .qos import QoSModel

class NetworkSlicingEnv(gym.Env):
    metadata = {"render_modes": []}
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.slice_configs = config.get('slices', [])
        self.n_slices = len(self.slice_configs)
        
        self.ep_length = config.get('episode_length', 288)
        self.tot_prbs = config.get('resources', {}).get('total_prbs', 100)
        self.tot_cus = config.get('resources', {}).get('total_cus', 50)
        
        self.traffic_gen = TrafficGenerator(self.slice_configs, self.ep_length, config.get('seed', 42))
        p_cfg = config.get('power', {})
        self.power_model = PowerModel(p_cfg.get('p_idle', 100), p_cfg.get('alpha', 200), p_cfg.get('beta', 100))
        self.qos_model = QoSModel(self.slice_configs)
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_slices * 7,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_slices * 2,), dtype=np.float32)
        
        self.w1 = config.get('reward', {}).get('w1', 1.0)
        self.w2 = config.get('reward', {}).get('w2', 0.3)
        self.w3 = config.get('reward', {}).get('w3', 2.0)
        
        self.step_idx = 0
        self.latency_recent = np.zeros(self.n_slices)
        self.violations_history = []
        self.last_demand_prb = np.zeros(self.n_slices)
        self.last_demand_cu = np.zeros(self.n_slices)
        self.last_alloc_prb = np.zeros(self.n_slices)
        self.last_alloc_cu = np.zeros(self.n_slices)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.traffic_gen.reset(seed)
        self.latency_recent = np.zeros(self.n_slices)
        self.violations_history = []
        
        self.last_demand_prb, self.last_demand_cu = self.traffic_gen.step(0)
        self.last_alloc_prb = np.full(self.n_slices, self.tot_prbs / self.n_slices)
        self.last_alloc_cu = np.full(self.n_slices, self.tot_cus / self.n_slices)
        
        return self._build_obs(), {}
        
    def _enforce_floors(self, alloc_fracs, total_res):
        floors = np.array([sc.get('floor_fraction', 0.0) for sc in self.slice_configs])
        allocs = alloc_fracs * total_res
        min_allocs = floors * total_res
        
        for i in range(self.n_slices):
            if allocs[i] < min_allocs[i]:
                allocs[i] = min_allocs[i]
                
        if np.sum(allocs) > total_res:
            surplus_needed = np.sum(allocs) - total_res
            non_floor_indices = np.where(allocs > min_allocs)[0]
            if len(non_floor_indices) > 0:
                non_floor_allocs = allocs[non_floor_indices]
                scale = max(0.0, 1.0 - surplus_needed / np.sum(non_floor_allocs))
                allocs[non_floor_indices] = allocs[non_floor_indices] * scale
                
        return allocs
        
    def step(self, action):
        action = np.clip(action, 0, 1)
        prb_logits = action[:self.n_slices]
        cu_logits = action[self.n_slices:]
        
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / max(e_x.sum(), 1e-6)
            
        prb_fracs = softmax(prb_logits)
        cu_fracs = softmax(cu_logits)
        
        alloc_prb = self._enforce_floors(prb_fracs, self.tot_prbs)
        alloc_cu = self._enforce_floors(cu_fracs, self.tot_cus)
        
        demand_prb, demand_cu = self.traffic_gen.step(self.step_idx)
        
        qos_res = self.qos_model.evaluate(demand_prb, demand_cu, alloc_prb, alloc_cu)
        
        self.latency_recent = 0.9 * self.latency_recent + 0.1 * qos_res['latency']
        self.violations_history.append(qos_res['violations'])
        if len(self.violations_history) > 10:
            self.violations_history.pop(0)
            
        prb_used = np.sum(qos_res['served_prb'])
        cu_used = np.sum(qos_res['served_cu'])
        energy = self.power_model.normalized_energy(prb_used, self.tot_prbs, cu_used, self.tot_cus)
        
        utilization = (prb_used / self.tot_prbs + cu_used / self.tot_cus) / 2.0
        
        viol_penalty = np.sum(qos_res['violations']) / self.n_slices
        reward = utilization - self.w1 * qos_res['dissatisfaction'] - self.w2 * energy - self.w3 * viol_penalty
        
        self.last_demand_prb = demand_prb
        self.last_demand_cu = demand_cu
        self.last_alloc_prb = alloc_prb
        self.last_alloc_cu = alloc_cu
        
        self.step_idx += 1
        terminated = False
        truncated = self.step_idx >= self.ep_length
        
        info = {
            'demand_prb': demand_prb, 'demand_cu': demand_cu,
            'alloc_prb': alloc_prb, 'alloc_cu': alloc_cu,
            'served_prb': qos_res['served_prb'], 'served_cu': qos_res['served_cu'],
            'latency': qos_res['latency'], 'violations': qos_res['violations'],
            'dissatisfaction': qos_res['dissatisfaction'], 'isolation_violations': qos_res['isolation_violations'],
            'energy': energy, 'utilization': utilization, 'step_idx': self.step_idx,
            'reward': reward
        }
        
        return self._build_obs(), float(reward), terminated, truncated, info
        
    def _build_obs(self):
        obs = np.zeros(self.n_slices * 7, dtype=np.float32)
        
        max_demand_prb = np.max([sc.get('base_demand_prb', 1) for sc in self.slice_configs]) * 3
        demand_est = np.clip(self.last_demand_prb / max(max_demand_prb, 1e-6), 0, 1)
        
        lat_norm = np.clip(self.latency_recent / 1000.0, 0, 1)
        
        if len(self.violations_history) > 0:
            viol_count = np.sum(self.violations_history, axis=0) / 10.0
        else:
            viol_count = np.zeros(self.n_slices)
            
        alloc_prb_frac = self.last_alloc_prb / self.tot_prbs
        alloc_cu_frac = self.last_alloc_cu / self.tot_cus
        
        priorities = np.array([sc.get('priority', 1) for sc in self.slice_configs])
        prio_norm = priorities / np.max(priorities)
        
        tod = (self.step_idx % self.ep_length) / self.ep_length
        tod_arr = np.full(self.n_slices, tod)
        
        n = self.n_slices
        obs[0:n] = demand_est
        obs[n:2*n] = lat_norm
        obs[2*n:3*n] = viol_count
        obs[3*n:4*n] = alloc_prb_frac
        obs[4*n:5*n] = alloc_cu_frac
        obs[5*n:6*n] = prio_norm
        obs[6*n:7*n] = tod_arr
        
        return obs
