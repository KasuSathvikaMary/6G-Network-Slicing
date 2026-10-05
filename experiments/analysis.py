#!/usr/bin/env python3
"""
6G Network Slicing - Results Analysis Tool

Analyzes experiment results, generates summary statistics,
and produces comparison reports.

Usage:
    python experiments/analysis.py --results-dir results/
    python experiments/analysis.py --results-dir results/ --format markdown
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes experiment results and generates reports."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        self.results: List[Dict[str, Any]] = []
        self._load_results()

    def _load_results(self):
        """Load all result files from the results directory."""
        if not os.path.exists(self.results_dir):
            logger.warning(f"Results directory '{self.results_dir}' does not exist.")
            return

        pattern = os.path.join(self.results_dir, "*.json")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} result files in {self.results_dir}")

        for filepath in sorted(files):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.results.extend(data)
                else:
                    self.results.append(data)
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")

    def get_experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific experiment."""
        exp_results = [r for r in self.results if r.get('experiment_name') == experiment_name]

        if not exp_results:
            return {"error": f"No results found for experiment '{experiment_name}'"}

        summary = {
            "experiment": experiment_name,
            "num_runs": len(exp_results),
            "timestamps": [r.get('timestamp', 'N/A') for r in exp_results],
            "avg_duration_s": float(np.mean([r.get('duration_seconds', 0) for r in exp_results])),
        }

        # Aggregate metrics across runs
        all_metrics = {}
        for r in exp_results:
            metrics = r.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        summary['metric_stats'] = {}
        for key, values in all_metrics.items():
            summary['metric_stats'][key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        return summary

    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall summary across all experiments."""
        experiment_names = set(r.get('experiment_name', 'unknown') for r in self.results)

        summaries = {}
        for name in sorted(experiment_names):
            summaries[name] = self.get_experiment_summary(name)

        return {
            "total_experiments": len(experiment_names),
            "total_runs": len(self.results),
            "experiment_summaries": summaries,
        }

    def compare_algorithms(self) -> Dict[str, Any]:
        """Compare algorithm performance from benchmark results."""
        benchmark_results = [
            r for r in self.results
            if 'algorithm' in r
        ]

        if not benchmark_results:
            return {"error": "No benchmark results found"}

        algorithms = set(r['algorithm'] for r in benchmark_results)
        scenarios = set(r.get('scenario', 'default') for r in benchmark_results)

        comparison = {}
        for algo in sorted(algorithms):
            algo_results = [r for r in benchmark_results if r['algorithm'] == algo]
            comparison[algo] = {
                'avg_reward': float(np.mean([r.get('avg_reward', 0) for r in algo_results])),
                'avg_qos': float(np.mean([r.get('avg_qos', 0) for r in algo_results])),
                'avg_energy': float(np.mean([r.get('avg_energy', 0) for r in algo_results])),
                'avg_latency': float(np.mean([r.get('avg_latency_ms', 0) for r in algo_results])),
                'sla_compliance': float(np.mean([r.get('sla_compliance', 0) for r in algo_results])),
                'fairness': float(np.mean([r.get('fairness_index', 0) for r in algo_results])),
                'inference_time_us': float(np.mean([r.get('inference_time_us', 0) for r in algo_results])),
                'num_runs': len(algo_results),
            }

        # Determine winners
        winners = {
            'best_reward': max(comparison.items(), key=lambda x: x[1]['avg_reward'])[0],
            'best_qos': max(comparison.items(), key=lambda x: x[1]['avg_qos'])[0],
            'lowest_energy': min(comparison.items(), key=lambda x: x[1]['avg_energy'])[0],
            'lowest_latency': min(comparison.items(), key=lambda x: x[1]['avg_latency'])[0],
            'best_sla': max(comparison.items(), key=lambda x: x[1]['sla_compliance'])[0],
            'best_fairness': max(comparison.items(), key=lambda x: x[1]['fairness'])[0],
            'fastest_inference': min(comparison.items(), key=lambda x: x[1]['inference_time_us'])[0],
        }

        return {
            'algorithms': comparison,
            'winners': winners,
            'scenarios': list(scenarios),
        }

    def generate_report(self, format: str = 'text') -> str:
        """Generate a comprehensive analysis report."""
        overall = self.get_overall_summary()

        if format == 'markdown':
            return self._generate_markdown_report(overall)
        else:
            return self._generate_text_report(overall)

    def _generate_text_report(self, summary: Dict) -> str:
        """Generate plain text report."""
        lines = [
            "=" * 80,
            "6G NETWORK SLICING - EXPERIMENT RESULTS ANALYSIS",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            f"Total Experiments: {summary['total_experiments']}",
            f"Total Runs: {summary['total_runs']}",
            "",
        ]

        for exp_name, exp_summary in summary.get('experiment_summaries', {}).items():
            lines.append(f"\n{'─' * 60}")
            lines.append(f"Experiment: {exp_name}")
            lines.append(f"  Runs: {exp_summary.get('num_runs', 0)}")
            lines.append(f"  Avg Duration: {exp_summary.get('avg_duration_s', 0):.2f}s")

            for metric, stats in exp_summary.get('metric_stats', {}).items():
                lines.append(
                    f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"[{stats['min']:.4f}, {stats['max']:.4f}]"
                )

        # Algorithm comparison
        algo_comparison = self.compare_algorithms()
        if 'algorithms' in algo_comparison:
            lines.append(f"\n{'=' * 80}")
            lines.append("ALGORITHM COMPARISON")
            lines.append(f"{'=' * 80}")

            for algo, metrics in algo_comparison['algorithms'].items():
                lines.append(f"\n  {algo}:")
                for metric, value in metrics.items():
                    lines.append(f"    {metric}: {value:.4f}")

            lines.append("\n  Winners:")
            for category, winner in algo_comparison.get('winners', {}).items():
                lines.append(f"    {category}: {winner}")

        return "\n".join(lines)

    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate Markdown formatted report."""
        lines = [
            "# 6G Network Slicing — Experiment Results Analysis",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|:---|:---|",
            f"| Total Experiments | {summary['total_experiments']} |",
            f"| Total Runs | {summary['total_runs']} |",
            "",
        ]

        # Per-experiment results
        lines.append("## Experiment Results\n")
        for exp_name, exp_summary in summary.get('experiment_summaries', {}).items():
            lines.append(f"### {exp_name}\n")
            lines.append(f"- **Runs:** {exp_summary.get('num_runs', 0)}")
            lines.append(f"- **Avg Duration:** {exp_summary.get('avg_duration_s', 0):.2f}s\n")

            metric_stats = exp_summary.get('metric_stats', {})
            if metric_stats:
                lines.append("| Metric | Mean | Std | Min | Max |")
                lines.append("|:---|:---:|:---:|:---:|:---:|")
                for metric, stats in metric_stats.items():
                    lines.append(
                        f"| {metric} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                        f"{stats['min']:.4f} | {stats['max']:.4f} |"
                    )
                lines.append("")

        # Algorithm comparison table
        algo_comparison = self.compare_algorithms()
        if 'algorithms' in algo_comparison:
            lines.append("## Algorithm Comparison\n")
            lines.append("| Algorithm | Reward | QoS | Energy | Latency(ms) | SLA% | Fairness |")
            lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

            for algo, metrics in algo_comparison['algorithms'].items():
                lines.append(
                    f"| {algo} | {metrics['avg_reward']:.4f} | {metrics['avg_qos']:.4f} | "
                    f"{metrics['avg_energy']:.4f} | {metrics['avg_latency']:.4f} | "
                    f"{metrics['sla_compliance']:.1%} | {metrics['fairness']:.4f} |"
                )

            lines.append("\n### Winners\n")
            for category, winner in algo_comparison.get('winners', {}).items():
                lines.append(f"- **{category}:** {winner}")

        return "\n".join(lines)

    def save_report(self, filepath: str, format: str = 'text'):
        """Save the analysis report to a file."""
        report = self.generate_report(format)
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Experiment Results Analysis")
    parser.add_argument(
        '--results-dir', type=str, default='results',
        help='Directory containing result JSON files'
    )
    parser.add_argument(
        '--format', type=str, default='text',
        choices=['text', 'markdown'],
        help='Report format'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: print to stdout)'
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        help='Analyze a specific experiment only'
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(args.results_dir)

    if args.experiment:
        summary = analyzer.get_experiment_summary(args.experiment)
        print(json.dumps(summary, indent=2, default=str))
    else:
        report = analyzer.generate_report(args.format)
        if args.output:
            analyzer.save_report(args.output, args.format)
        else:
            print(report)


if __name__ == "__main__":
    main()
