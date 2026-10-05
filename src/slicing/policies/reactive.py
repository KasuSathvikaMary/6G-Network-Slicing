"""Reactive proportional policy - allocates proportionally to recent demand."""

import numpy as np
from .base import BasePolicy


class ReactivePolicy(BasePolicy):
    """Allocates resources proportionally to the last observed demand.
    
    Uses exponential moving average (EMA) smoothing of demand estimates
    from the observation vector. Falls back to equal allocation when
    all demands are near zero.
    """
    
    def __init__(self, config: dict, n_slices: int, n_resources: int = 2):
        super().__init__(config, n_slices, n_resources)
        self.alpha = config.get('reactive', {}).get('smoothing_alpha', 0.3)
        self._smoothed_demand = None
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Allocate proportionally to EMA-smoothed demand estimates."""
        # Extract demand estimates from obs (first n_slices elements)
        current_demand = obs[:self.n_slices].copy()
        
        if self._smoothed_demand is None:
            self._smoothed_demand = current_demand.copy()
        else:
            self._smoothed_demand = (self.alpha * current_demand + 
                                     (1 - self.alpha) * self._smoothed_demand)
        
        # Allocate proportionally to smoothed demand
        total = np.sum(self._smoothed_demand)
        if total < 1e-8:
            fracs = np.ones(self.n_slices) / self.n_slices
        else:
            fracs = self._smoothed_demand / total
        
        # Same fractions for PRB and CU
        action = np.concatenate([fracs, fracs])
        return action
    
    @property
    def name(self) -> str:
        return "reactive"
