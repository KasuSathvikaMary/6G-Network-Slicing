"""Oracle policy - hindsight optimal allocation with perfect demand knowledge.

This is an upper-bound reference: it peeks at the true demand from the
traffic generator and allocates optimally. Not a fair comparison with
other policies, but useful for gauging headroom.
"""

import numpy as np
from .base import BasePolicy


class OraclePolicy(BasePolicy):
    """Hindsight-optimal policy using perfect demand knowledge.
    
    Requires a reference to the environment to peek at true demand.
    Allocates proportionally to priority × demand, ensuring resources
    go where they're needed most.
    """
    
    def __init__(self, config: dict, n_slices: int, n_resources: int = 2, env=None):
        super().__init__(config, n_slices, n_resources)
        self.env = env
        self.priorities = np.array([
            s.get('priority', 1) for s in config.get('slices', [])
        ], dtype=np.float64)
        if len(self.priorities) < n_slices:
            self.priorities = np.ones(n_slices)
    
    def set_env(self, env):
        """Set environment reference for demand peeking."""
        self.env = env
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Allocate optimally using true demand from the environment."""
        if self.env is None:
            # Fallback: use demand estimates from obs
            demand = obs[:self.n_slices].copy()
        else:
            # Peek at true demand
            demand_prb, demand_cu = self.env.traffic_gen.step(self.env.step_idx)
            # Normalize demands to fractions of total capacity
            total_prb = self.env.tot_prbs
            total_cu = self.env.tot_cus
            
            # Priority-weighted allocation for PRBs
            weighted_prb = self.priorities * demand_prb
            total_w_prb = np.sum(weighted_prb)
            if total_w_prb > 1e-8:
                prb_fracs = weighted_prb / total_w_prb
            else:
                prb_fracs = np.ones(self.n_slices) / self.n_slices
            
            # Priority-weighted allocation for CUs
            weighted_cu = self.priorities * demand_cu
            total_w_cu = np.sum(weighted_cu)
            if total_w_cu > 1e-8:
                cu_fracs = weighted_cu / total_w_cu
            else:
                cu_fracs = np.ones(self.n_slices) / self.n_slices
            
            return np.concatenate([prb_fracs, cu_fracs])
        
        # Fallback path using obs
        weighted = self.priorities * demand
        total = np.sum(weighted)
        if total > 1e-8:
            fracs = weighted / total
        else:
            fracs = np.ones(self.n_slices) / self.n_slices
        return np.concatenate([fracs, fracs])
    
    @property
    def name(self) -> str:
        return "oracle"
