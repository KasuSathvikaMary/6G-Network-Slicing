import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from slicing.ops.audit import AuditLog

def test_append_and_verify():
    log = AuditLog()
    for i in range(10):
        obs = np.random.rand(10)
        action = np.random.rand(2)
        reward = np.random.rand()
        log.append(i, obs, action, reward)
        
    is_valid, msg = log.verify()
    assert is_valid == True
    
def test_tamper_detection():
    log = AuditLog()
    for i in range(5):
        obs = np.random.rand(10)
        action = np.random.rand(2)
        reward = np.random.rand()
        log.append(i, obs, action, reward)
        
    log.entries[2]['action'] = [0.0, 0.0] # tamper
    is_valid, msg = log.verify()
    assert is_valid == False
    
def test_empty_log():
    log = AuditLog()
    is_valid, msg = log.verify()
    assert is_valid == True
    
def test_chain_integrity():
    log = AuditLog()
    for i in range(5):
        obs = np.random.rand(10)
        action = np.random.rand(2)
        reward = np.random.rand()
        log.append(i, obs, action, reward)
        
    # Break chain by modifying prev_hash
    log.entries[3]['prev_hash'] = '1'*64
    is_valid, msg = log.verify()
    assert is_valid == False
