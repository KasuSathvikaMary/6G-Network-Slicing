"""Abstract base class for network slicing allocation policies."""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """Base policy interface for network slice resource allocation.
    
    All policies must implement select_action() and the name property.
    Training, saving, and loading are optional capabilities.
    """
    
    def __init__(self, config: dict, n_slices: int, n_resources: int = 2):
        self.config = config
        self.n_slices = n_slices
        self.n_resources = n_resources
    
    @abstractmethod
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Select allocation action given observation.
        
        Args:
            obs: Observation vector from the environment.
            
        Returns:
            Action array of shape (n_slices * n_resources,).
            First n_slices values are PRB fractions, next n_slices are CU fractions.
            Values will be normalized by the environment.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this policy."""
        pass
    
    def train(self, env, total_timesteps: int, **kwargs) -> dict:
        """Train the policy (optional). Returns training info dict."""
        return {}
    
    def save(self, path: str) -> None:
        """Save policy to disk (optional)."""
        pass
    
    def load(self, path: str) -> None:
        """Load policy from disk (optional)."""
        pass
