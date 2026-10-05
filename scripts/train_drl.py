#!/usr/bin/env python3
import argparse
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from slicing.utils import load_config, set_seed
from slicing.env import NetworkSlicingEnv
from slicing.policies.drl_policy import DRLPolicy

def main():
    parser = argparse.ArgumentParser(description="Train DRL policy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="models/")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)
    
    env = NetworkSlicingEnv(config)
    n_slices = len(config.get("slices", []))
    
    policy = DRLPolicy(config, n_slices=n_slices)
    timesteps = args.timesteps if args.timesteps is not None else config.get("drl", {}).get("total_timesteps", 100000)
    
    print(f"Starting training for {timesteps} timesteps...")
    t0 = time.perf_counter()
    train_info = policy.train(env, total_timesteps=timesteps)
    t1 = time.perf_counter()
    
    print(f"Training completed in {t1-t0:.2f} seconds.")
    
    model_path = os.path.join(args.output, "best_model.zip")
    policy.save(model_path)
    
    df = pd.DataFrame(train_info if isinstance(train_info, dict) else [{"time": t1-t0}])
    df.to_csv(os.path.join(args.output, "training_log.csv"), index=False)
    
if __name__ == "__main__":
    main()
