"""DRL policy using PPO from stable-baselines3.

Inspired by Works 1 (DRL for resource management, 2022) and 4 (Energy-aware 
actor-critic, 2024): uses PPO with energy-aware reward for dynamic allocation.
"""

import os
import time
import numpy as np
from .base import BasePolicy


class DRLPolicy(BasePolicy):
    """PPO-based deep reinforcement learning policy for network slicing.
    
    Uses stable-baselines3 PPO with configurable architecture.
    The energy term in the reward (handled by the environment) addresses
    the energy-awareness aspect from Work 4.
    """
    
    def __init__(self, config: dict, n_slices: int, n_resources: int = 2):
        super().__init__(config, n_slices, n_resources)
        self.model = None
        self.drl_config = config.get('drl', {})
        self._trained = False
    
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        """Use trained PPO model to select action."""
        if self.model is None:
            raise RuntimeError("DRL model not trained or loaded. Call train() or load() first.")
        action, _ = self.model.predict(obs, deterministic=True)
        return action
    
    def train(self, env, total_timesteps: int = None, **kwargs) -> dict:
        """Train PPO model on the environment.
        
        Args:
            env: Gymnasium environment
            total_timesteps: Override training steps (uses config if None)
            
        Returns:
            Dict with training_time and total_timesteps
        """
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        
        if total_timesteps is None:
            total_timesteps = self.drl_config.get('total_timesteps', 200000)
        
        net_arch = self.drl_config.get('net_arch', [64, 64])
        
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.drl_config.get('learning_rate', 3e-4),
            gamma=self.drl_config.get('gamma', 0.99),
            n_steps=self.drl_config.get('n_steps', 2048),
            batch_size=self.drl_config.get('batch_size', 64),
            n_epochs=self.drl_config.get('n_epochs', 10),
            clip_range=self.drl_config.get('clip_range', 0.2),
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
            seed=self.config.get('seed', 42),
        )
        
        start_time = time.time()
        self.model.learn(total_timesteps=total_timesteps)
        training_time = time.time() - start_time
        
        self._trained = True
        
        return {
            'training_time': training_time,
            'total_timesteps': total_timesteps,
        }
    
    def save(self, path: str) -> None:
        """Save trained model to disk."""
        if self.model is not None:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            self.model.save(path)
    
    def load(self, path: str) -> None:
        """Load trained model from disk."""
        from stable_baselines3 import PPO
        self.model = PPO.load(path)
        self._trained = True
    
    @property
    def name(self) -> str:
        return "drl_ppo"
