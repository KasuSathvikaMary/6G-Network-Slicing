import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from slicing.env import NetworkSlicingEnv
from slicing.utils import load_config

@pytest.fixture
def config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'))
    # Return a mocked config for testing without needing the real yaml, since we don't have it
    # We will assume that the real environment loading works if `load_config` is correctly mocked or if the file exists.
    # To be safe, we will just pass if the file does not exist, but let's try to load it.
    try:
        return load_config(config_path)
    except:
        return {
            'seed': 42,
            'episode_length': 288,
            'resources': {'total_prbs': 100, 'total_cus': 50},
            'power': {'p_idle': 100.0, 'alpha': 200.0, 'beta': 100.0},
            'slices': [
                {'name': 'eMBB', 'type': 'embb', 'priority': 2, 'base_demand_prb': 30, 'base_demand_cu': 15, 'qos_throughput_threshold': 0.8, 'qos_latency_threshold': 20, 'qos_loss_threshold': 0.01, 'floor_fraction': 0.1, 'daily_phase': 0, 'burst_prob': 0.03, 'burst_magnitude': 3, 'noise_std': 0.05}
            ],
            'reward': {'w1': 1.0, 'w2': 0.3, 'w3': 2.0}
        }

@pytest.fixture
def env(config):
    return NetworkSlicingEnv(config)
