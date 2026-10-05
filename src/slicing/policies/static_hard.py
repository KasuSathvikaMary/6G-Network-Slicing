"""Static hard partitioning policy - equal fixed allocation."""

import numpy as np
from .base import BasePolicy


class StaticHardPolicy(BasePolicy):
    """Allocates equal fixed fractions to each slice.
    
    This is the simplest baseline: each slice gets 1/n_slices of each resource
    regardless of demand. Represents traditional static network slicing.
    """
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Return equal fractions for all slices and resources."""
        action = np.ones(self.n_slices * self.n_resources) / self.n_slices
        return action
    
    @property
    def name(self) -> str:
        return "static_hard"
