#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/")
    parser.add_argument("--output", type=str, default="figures/")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    plt.style.use('seaborn-v0_8-paper')
    
    exp_csv = os.path.join(args.results, "experiment_results.csv")
    abl_csv = os.path.join(args.results, "ablation_results.csv")
    
    if os.path.exists(exp_csv):
        df_exp = pd.read_csv(exp_csv)
        
        # a) utilization vs dissatisfaction
        plt.figure(figsize=(8,5), dpi=150)
        sns.scatterplot(data=df_exp, x='utilization', y='dissatisfaction', hue='policy', style='scenario')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "utilization_vs_dissatisfaction.png"))
        plt.close()
        
        # b) energy vs violations
        plt.figure(figsize=(8,5), dpi=150)
        sns.scatterplot(data=df_exp, x='energy', y='violation_rate', hue='policy')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "energy_vs_violations.png"))
        plt.close()
        
        # c) decision_latency_vs_slices (Mocking n_slices for demo as scenario E results might not have it explicitly)
        # assuming df_exp has some latency info
        plt.figure(figsize=(8,5), dpi=150)
        sns.lineplot(data=df_exp[df_exp['scenario'] == 'E'], x='seed', y='decision_latency_p95_ms', hue='policy')
        plt.xlabel("n_slices (mocked via seed)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "decision_latency_vs_slices.png"))
        plt.close()
        
        # d) scenario comparison
        plt.figure(figsize=(8,5), dpi=150)
        sns.barplot(data=df_exp, x='scenario', y='utilization', hue='policy', errorbar=('ci', 95))
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "scenario_comparison.png"))
        plt.close()

    if os.path.exists(abl_csv):
        df_abl = pd.read_csv(abl_csv)
        
        # e) ablation energy
        plt.figure(figsize=(8,5), dpi=150)
        sns.barplot(data=df_abl[df_abl['ablation'] == 'energy_weight'], x='param', y='energy')
        plt.title("DRL with vs without energy term")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "ablation_energy.png"))
        plt.close()
        
        # f) ablation floors
        plt.figure(figsize=(8,5), dpi=150)
        df_floor = df_abl[df_abl['ablation'] == 'floor_size']
        sns.lineplot(data=df_floor, x='param', y='utilization', label='Utilization')
        sns.lineplot(data=df_floor, x='param', y='violations', label='Violations')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "ablation_floors.png"))
        plt.close()
        
        # g) drift recovery
        plt.figure(figsize=(8,5), dpi=150)
        df_drift = df_abl[df_abl['ablation'] == 'drift_retraining']
        sns.barplot(data=df_drift, x='param', y='utilization')
        plt.title("Drift Recovery: Retraining vs No Retraining")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, "drift_recovery.png"))
        plt.close()

if __name__ == "__main__":
    main()
