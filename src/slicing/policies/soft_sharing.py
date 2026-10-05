"""Soft sharing policy - guaranteed floors with priority-weighted surplus sharing.

Inspired by Work 6 (Dynamic soft slicing, 2025): provides isolation through
guaranteed minimums while improving utilization by sharing unused capacity
based on priority-weighted shortfalls.
"""

import numpy as np
from .base import BasePolicy


class SoftSharingPolicy(BasePolicy):
    """Guarantees a floor allocation per slice, shares surplus by priority × shortfall.
    
    Step 1: Allocate floor to each slice
    Step 2: Compute surplus = 1.0 - sum(floors)
    Step 3: For each slice, shortfall = max(0, demand_est - floor)
    Step 4: weighted_need = priority * shortfall
    Step 5: Distribute surplus proportionally to weighted_need
    """
    
    def __init__(self, config: dict, n_slices: int, n_resources: int = 2):
        super().__init__(config, n_slices, n_resources)
        self.floors = np.array([
            s.get('floor_fraction', 0.1) for s in config.get('slices', [])
        ])
        self.priorities = np.array([
            s.get('priority', 1) for s in config.get('slices', [])
        ], dtype=np.float64)
        # Handle case where config has fewer slices than n_slices
        if len(self.floors) < n_slices:
            self.floors = np.full(n_slices, 0.1)
            self.priorities = np.ones(n_slices)
    
    def _allocate_resource(self, demand_est: np.ndarray) -> np.ndarray:
        """Allocate one resource type with floor guarantee + priority sharing."""
        # Step 1: Start with floors
        alloc = self.floors.copy()
        
        # Step 2: Surplus capacity
        surplus = max(0.0, 1.0 - np.sum(alloc))
        
        # Step 3: Shortfall per slice
        shortfall = np.maximum(0.0, demand_est - alloc)
        
        # Step 4: Priority-weighted need
        weighted_need = self.priorities * shortfall
        total_weighted = np.sum(weighted_need)
        
        # Step 5: Distribute surplus proportionally
        if total_weighted > 1e-8 and surplus > 1e-8:
            alloc += surplus * (weighted_need / total_weighted)
        elif surplus > 1e-8:
            # No shortfall - distribute surplus equally
            alloc += surplus / self.n_slices
        
        return alloc
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select action with floor guarantees and priority-weighted sharing."""
        # Extract demand estimates from obs
        demand_est = obs[:self.n_slices].copy()
        
        # Allocate PRBs and CUs independently
        prb_fracs = self._allocate_resource(demand_est)
        cu_fracs = self._allocate_resource(demand_est)
        
        return np.concatenate([prb_fracs, cu_fracs])
    
    @property
    def name(self) -> str:
        return "soft_sharing"
