import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicing.env import NetworkSlicingEnv
from slicing.utils import load_config

def test_obs_shape(env):
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape

def test_obs_bounds(env):
    obs, _ = env.reset(seed=42)
    assert np.all(obs >= 0) and np.all(obs <= 1.0 + 1e-6)

def test_allocation_never_exceeds_capacity(env, config):
    obs, _ = env.reset(seed=42)
    total_prbs = config['resources']['total_prbs']
    total_cus = config['resources']['total_cus']
    for _ in range(50):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert info['alloc_prb'].sum() <= total_prbs + 1e-6
        assert info['alloc_cu'].sum() <= total_cus + 1e-6

def test_episode_length(env, config):
    obs, _ = env.reset(seed=42)
    for i in range(config['episode_length']):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
    assert trunc == True

def test_reward_components(env):
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    assert 'utilization' in info
    assert 'dissatisfaction' in info
    assert 'energy' in info
