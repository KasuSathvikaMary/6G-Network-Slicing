import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicing.env import NetworkSlicingEnv
from slicing.utils import load_config

def run_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    trajectory = []
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        trajectory.append((obs.copy(), reward))
        if term or trunc:
            break
            
    return trajectory

def test_same_seed_same_trajectory(config):
    env1 = NetworkSlicingEnv(config)
    env2 = NetworkSlicingEnv(config)
    
    traj1 = run_episode(env1, seed=42)
    traj2 = run_episode(env2, seed=42)
    
    assert len(traj1) == len(traj2)
    for (obs1, r1), (obs2, r2) in zip(traj1, traj2):
        np.testing.assert_array_almost_equal(obs1, obs2)
        assert r1 == r2

def test_different_seed_different_trajectory(config):
    env1 = NetworkSlicingEnv(config)
    env2 = NetworkSlicingEnv(config)
    
    traj1 = run_episode(env1, seed=42)
    traj2 = run_episode(env2, seed=99)
    
    # Very unlikely to be exactly the same
    are_different = False
    for (obs1, r1), (obs2, r2) in zip(traj1, traj2):
        if not np.allclose(obs1, obs2) or r1 != r2:
            are_different = True
            break
            
    assert are_different
