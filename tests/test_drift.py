import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicing.ops.drift import DriftDetector

def test_no_drift_same_distribution():
    ref_data = np.random.normal(0, 1, size=(1000, 5))
    detector = DriftDetector(ref_data, window_size=50)
    
    test_data = np.random.normal(0, 1, size=(100, 5))
    results = []
    for obs in test_data:
        results.append(detector.update(obs))
        
    # The last few should have psi < threshold
    assert results[-1]['drift_detected'] == False

def test_drift_detected():
    ref_data = np.random.normal(0, 1, size=(1000, 5))
    detector = DriftDetector(ref_data, window_size=50)
    
    test_data = np.random.normal(10, 2, size=(100, 5))
    results = []
    for obs in test_data:
        results.append(detector.update(obs))
        
    assert results[-1]['drift_detected'] == True

def test_reset():
    ref_data = np.random.normal(0, 1, size=(1000, 5))
    detector = DriftDetector(ref_data, window_size=50)
    
    detector.update(np.random.normal(0, 1, size=(5,)))
    assert len(detector.buffer) == 1
    detector.reset()
    assert len(detector.buffer) == 0

def test_psi_computation():
    ref_data = np.random.normal(0, 1, size=(1000, 1))
    detector = DriftDetector(ref_data, window_size=10)
    # Just checking it doesn't crash and returns a float
    psi = detector._compute_psi(np.random.normal(0, 1, size=(10, 1)))
    assert isinstance(psi, float)
