import pytest
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicing.metrics import MetricsCollector

def test_metrics_collector():
    collector = MetricsCollector()
    collector.record_step({'utilization': 0.8, 'dissatisfaction': 0.1, 'energy': 150.0, 'reward': 10.0, 'violations': [False], 'isolation_violations': 0})
    metrics = collector.get_episode_metrics()
    
    assert 'utilization_mean' in metrics
    assert 'violation_rate' in metrics
    assert 'dissatisfaction_mean' in metrics
    assert 'isolation_violations_total' in metrics
    assert 'energy_mean' in metrics
    assert 'reward_total' in metrics
    assert 'reward_mean' in metrics
    
    df = collector.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert 'utilization' in df.columns
    
    collector.reset()
    assert len(collector.to_dataframe()) == 0
