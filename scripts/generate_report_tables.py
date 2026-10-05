#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/")
    parser.add_argument("--output", type=str, default="docs/")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    exp_csv = os.path.join(args.results, "experiment_results.csv")
    abl_csv = os.path.join(args.results, "ablation_results.csv")
    
    out_md = os.path.join(args.output, "tables.md")
    
    with open(out_md, 'w') as f:
        f.write("# Experiment Results\n\n")
        
        if os.path.exists(exp_csv):
            df = pd.read_csv(exp_csv)
            
            # Table 1
            f.write("## Table 1: Main Results per Scenario x Policy\n\n")
            agg_df = df.groupby(['scenario', 'policy']).agg({
                'utilization': ['mean', 'std'],
                'violation_rate': ['mean', 'std']
            }).reset_index()
            # Simple markdown table format
            f.write("| Scenario | Policy | Utilization (Mean ± Std) | Violation Rate (Mean ± Std) |\n")
            f.write("|---|---|---|---|\n")
            for _, row in agg_df.iterrows():
                f.write(f"| {row['scenario'].values[0]} | {row['policy'].values[0]} | {row['utilization']['mean']:.3f} ± {row['utilization']['std']:.3f} | {row['violation_rate']['mean']:.3f} ± {row['violation_rate']['std']:.3f} |\n")
            f.write("\n")
            
            # Table 2: Statistical tests
            f.write("## Table 2: Statistical Tests (Mann-Whitney U p-values)\n\n")
            policies = df['policy'].unique()
            f.write("| Scenario | Policy A | Policy B | p-value |\n")
            f.write("|---|---|---|---|\n")
            for scenario in df['scenario'].unique():
                scen_df = df[df['scenario'] == scenario]
                for i in range(len(policies)):
                    for j in range(i+1, len(policies)):
                        pA = policies[i]
                        pB = policies[j]
                        dataA = scen_df[scen_df['policy'] == pA]['utilization']
                        dataB = scen_df[scen_df['policy'] == pB]['utilization']
                        if len(dataA) > 0 and len(dataB) > 0:
                            stat, pval = mannwhitneyu(dataA, dataB, alternative='two-sided')
                            f.write(f"| {scenario} | {pA} | {pB} | {pval:.4e} |\n")
            f.write("\n")
            
        if os.path.exists(abl_csv):
            df_abl = pd.read_csv(abl_csv)
            f.write("## Table 3: Ablation Results\n\n")
            f.write("| Ablation | Param | Utilization | Violations | Energy |\n")
            f.write("|---|---|---|---|---|\n")
            for _, row in df_abl.iterrows():
                f.write(f"| {row['ablation']} | {row['param']} | {row['utilization']:.3f} | {row['violations']:.3f} | {row['energy']:.3f} |\n")
            f.write("\n")
            
        f.write("## Table 4: Explanation and Audit Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write("| Avg SHAP computation time (ms) | TBD |\n")
        f.write("| Audit log consensus overhead (ms) | TBD |\n")

if __name__ == "__main__":
    main()
