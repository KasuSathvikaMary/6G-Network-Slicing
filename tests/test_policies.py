import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from slicing.policies.static_hard import StaticHardPolicy
from slicing.policies.reactive import ReactivePolicy
from slicing.policies.soft_sharing import SoftSharingPolicy

def test_static_hard_policy(config, env):
    n_slices = len(config['slices'])
    policy = StaticHardPolicy(config, n_slices)
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs)
    
    assert action.shape == env.action_space.shape
    assert policy.name == "StaticHard"
    # test capacity
    assert np.sum(action[:n_slices]) <= 1.0 + 1e-6
    assert np.sum(action[n_slices:]) <= 1.0 + 1e-6

def test_reactive_policy(config, env):
    n_slices = len(config['slices'])
    policy = ReactivePolicy(config, n_slices)
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs)
    
    assert action.shape == env.action_space.shape
    assert policy.name == "Reactive"
    assert np.sum(action[:n_slices]) <= 1.0 + 1e-6
    assert np.sum(action[n_slices:]) <= 1.0 + 1e-6

def test_soft_sharing_policy(config, env):
    n_slices = len(config['slices'])
    policy = SoftSharingPolicy(config, n_slices)
    obs, _ = env.reset(seed=42)
    action = policy.select_action(obs)
    
    assert action.shape == env.action_space.shape
    assert policy.name == "SoftSharing"
    assert np.sum(action[:n_slices]) <= 1.0 + 1e-6
    assert np.sum(action[n_slices:]) <= 1.0 + 1e-6
