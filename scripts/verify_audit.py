#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slicing.ops.audit import AuditLog

def main():
    print("--- 6G Network Slicing: Audit Log Verification ---")
    audit = AuditLog(consensus_delay_ms=0.0)
    
    # 1. Append valid entries
    print("Appending valid entries...")
    audit.append(step=0, obs=[0.1, 0.2], action=[0.5, 0.5], reward=1.0)
    audit.append(step=1, obs=[0.3, 0.4], action=[0.6, 0.4], reward=0.8)
    audit.append(step=2, obs=[0.5, 0.6], action=[0.7, 0.3], reward=0.9)
    
    # 2. Verify
    is_valid, msg = audit.verify()
    print(f"Verification (Expected True): {is_valid} - {msg}")
    
    # 3. Tamper
    print("Tampering with entry 1...")
    if len(audit.chain) > 1:
        audit.chain[1]['reward'] = 999.9  # tampering
        
    # 4. Verify again
    is_valid, msg = audit.verify()
    print(f"Verification (Expected False): {is_valid} - {msg}")

if __name__ == "__main__":
    main()
