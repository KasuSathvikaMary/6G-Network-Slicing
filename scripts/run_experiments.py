#!/usr/bin/env python3
import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slicing.utils import load_config, set_seed, merge_configs
from slicing.env import NetworkSlicingEnv
from slicing.metrics import MetricsCollector
from slicing.policies.static_hard import StaticHardPolicy
from slicing.policies.reactive import ReactivePolicy
from slicing.policies.soft_sharing import SoftSharingPolicy
from slicing.policies.drl_policy import DRLPolicy
from slicing.policies.oracle import OraclePolicy
from slicing.ops.audit import AuditLog
from slicing.ops.explainability import explain_decision

def get_policy(name, config, n_slices, env):
    if name == "static":
        return StaticHardPolicy(config, n_slices)
    elif name == "reactive":
        return ReactivePolicy(config, n_slices)
    elif name == "soft":
        return SoftSharingPolicy(config, n_slices)
    elif name == "drl":
        policy = DRLPolicy(config, n_slices)
        model_path = os.path.join("models", "best_model.zip")
        if os.path.exists(model_path):
            policy.load(model_path)
        return policy
    elif name == "oracle":
        return OraclePolicy(config, n_slices, env=env)
    raise ValueError(f"Unknown policy: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--policies", type=str, default="all")
    parser.add_argument("--n-seeds", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    base_config = load_config(args.config)
    
    scenarios = ["A", "B", "C", "D", "E"] if args.scenarios == "all" else args.scenarios.split(",")
    policies = ["static", "reactive", "soft", "drl", "oracle"] if args.policies == "all" else args.policies.split(",")
    n_seeds = args.n_seeds if args.n_seeds is not None else base_config.get("experiment", {}).get("n_seeds", 1)
    
    results = []
    
    for scenario in scenarios:
        scenario_config_path = f"configs/scenario_{scenario}.yaml"
        if os.path.exists(scenario_config_path):
            scenario_config = load_config(scenario_config_path)
            config = merge_configs(base_config, scenario_config)
        else:
            config = base_config
            
        env = NetworkSlicingEnv(config)
        n_slices = len(config.get("slices", []))
        
        for policy_name in policies:
            print(f"Running Scenario {scenario} with Policy {policy_name}")
            policy = get_policy(policy_name, config, n_slices, env)
            
            for seed in range(n_seeds):
                set_seed(seed)
                obs, _ = env.reset(seed=seed)
                done = False
                metrics = MetricsCollector()
                audit = AuditLog(consensus_delay_ms=config.get("audit", {}).get("consensus_delay_ms", 0.0))
                
                latencies = []
                steps = 0
                bg_data = []
                
                while not done:
                    t0 = time.perf_counter()
                    action = policy.select_action(obs)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    metrics.record_step(info)
                    audit.append(steps, obs, action, reward)
                    
                    if len(bg_data) < 100:
                        bg_data.append(obs)
                    
                    obs = next_obs
                    steps += 1
                
                ep_metrics = metrics.get_episode_metrics()
                latencies_arr = np.array(latencies)
                
                row = {
                    "scenario": scenario,
                    "policy": policy_name,
                    "seed": seed,
                    "utilization": ep_metrics.get("utilization_mean", 0),
                    "violation_rate": ep_metrics.get("violation_rate", 0),
                    "dissatisfaction": ep_metrics.get("dissatisfaction_mean", 0),
                    "isolation_violations": ep_metrics.get("isolation_violations_total", 0),
                    "energy": ep_metrics.get("energy_mean", 0),
                    "reward_mean": ep_metrics.get("reward_mean", 0),
                    "decision_latency_p50_ms": np.percentile(latencies_arr, 50) if len(latencies_arr)>0 else 0,
                    "decision_latency_p95_ms": np.percentile(latencies_arr, 95) if len(latencies_arr)>0 else 0,
                }
                results.append(row)
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output, "experiment_results.csv"), index=False)

if __name__ == "__main__":
    main()
