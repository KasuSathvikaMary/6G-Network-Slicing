#!/usr/bin/env python3
"""
6G Network Slicing - DRL Benchmarking Tool

Benchmarks the Deep Reinforcement Learning engine across different
configurations and compares against baseline algorithms.

Usage:
    python experiments/benchmark.py --episodes 2000 --scenario normal
    python experiments/benchmark.py --compare-all
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Stores benchmark results for a single algorithm run."""
    algorithm: str
    scenario: str
    episodes: int
    avg_reward: float
    std_reward: float
    avg_qos: float
    avg_energy: float
    avg_latency_ms: float
    convergence_episode: int
    inference_time_us: float  # microseconds
    training_time_s: float
    memory_mb: float
    sla_compliance: float
    fairness_index: float


class BaselineAlgorithm:
    """Base class for baseline resource allocation algorithms."""

    def __init__(self, num_slices: int, total_resources: np.ndarray):
        self.num_slices = num_slices
        self.total_resources = total_resources  # [bandwidth, cpu, memory, energy]

    def allocate(self, state: np.ndarray, step: int) -> np.ndarray:
        raise NotImplementedError


class StaticAllocation(BaselineAlgorithm):
    """Equal static allocation across all slices."""

    def allocate(self, state: np.ndarray, step: int) -> np.ndarray:
        per_slice = self.total_resources / self.num_slices
        return np.tile(per_slice, (self.num_slices, 1))


class RoundRobinAllocation(BaselineAlgorithm):
    """Round-robin allocation with priority rotation."""

    def allocate(self, state: np.ndarray, step: int) -> np.ndarray:
        allocation = np.zeros((self.num_slices, 4))
        priority_order = [(step + i) % self.num_slices for i in range(self.num_slices)]

        remaining = self.total_resources.copy()
        for rank, slice_idx in enumerate(priority_order):
            share = remaining / (self.num_slices - rank)
            weight = 1.0 + 0.3 * (self.num_slices - rank) / self.num_slices
            allocation[slice_idx] = share * weight
            remaining -= allocation[slice_idx]
            remaining = np.maximum(remaining, 0)

        return allocation


class ProportionalFairAllocation(BaselineAlgorithm):
    """Proportional fair allocation based on demand."""

    def __init__(self, num_slices: int, total_resources: np.ndarray):
        super().__init__(num_slices, total_resources)
        self.avg_throughput = np.ones((num_slices, 4))

    def allocate(self, state: np.ndarray, step: int) -> np.ndarray:
        demand = np.abs(state.reshape(self.num_slices, -1)[:, :4]) + 0.1
        demand = demand[:, :4] if demand.shape[1] >= 4 else np.pad(
            demand, ((0, 0), (0, 4 - demand.shape[1]))
        )

        # Proportional fair metric: demand / average_throughput
        pf_metric = demand / (self.avg_throughput + 1e-8)
        weights = pf_metric / (pf_metric.sum(axis=0, keepdims=True) + 1e-8)

        allocation = weights * self.total_resources[np.newaxis, :]

        # Update average throughput (exponential moving average)
        alpha = 0.1
        self.avg_throughput = (1 - alpha) * self.avg_throughput + alpha * allocation

        return allocation


class GreedyAllocation(BaselineAlgorithm):
    """Greedy allocation maximizing immediate QoS."""

    def allocate(self, state: np.ndarray, step: int) -> np.ndarray:
        # Allocate most resources to highest-demand slices
        demand = np.abs(state.reshape(self.num_slices, -1)[:, :4]) + 0.1
        demand = demand[:, :4] if demand.shape[1] >= 4 else np.pad(
            demand, ((0, 0), (0, 4 - demand.shape[1]))
        )

        sorted_indices = np.argsort(-demand.sum(axis=1))
        allocation = np.zeros((self.num_slices, 4))
        remaining = self.total_resources.copy()

        for idx in sorted_indices:
            give = np.minimum(demand[idx] * 1.2, remaining)
            allocation[idx] = give
            remaining -= give
            remaining = np.maximum(remaining, 0)

        return allocation


class DRLBenchmark:
    """
    Benchmarking framework for DRL resource allocation.
    Simulates the environment and compares algorithms.
    """

    def __init__(
        self,
        num_slices: int = 10,
        total_resources: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        self.num_slices = num_slices
        self.total_resources = total_resources if total_resources is not None else \
            np.array([1000.0, 100.0, 512.0, 5000.0])  # bandwidth_MHz, cpu_cores, memory_GB, energy_W
        self.seed = seed
        np.random.seed(seed)

        self.algorithms: Dict[str, BaselineAlgorithm] = {
            'static': StaticAllocation(num_slices, self.total_resources),
            'round_robin': RoundRobinAllocation(num_slices, self.total_resources),
            'proportional_fair': ProportionalFairAllocation(num_slices, self.total_resources),
            'greedy': GreedyAllocation(num_slices, self.total_resources),
        }

    def generate_scenario(self, scenario: str, step: int, total_steps: int) -> np.ndarray:
        """Generate demand state for a given scenario."""
        t = step / total_steps  # Normalized time

        if scenario == 'normal':
            demand = np.random.exponential(0.5, (self.num_slices, 10))
            demand[:, 0] *= 100  # bandwidth
            demand[:, 1] *= 0.5  # latency
            demand[:, 2] *= 10   # compute
            demand[:, 3] *= 50   # memory

        elif scenario == 'peak_hour':
            # Sinusoidal pattern with peak
            peak_factor = 1.0 + 1.5 * np.sin(2 * np.pi * t)
            demand = np.random.exponential(0.5 * peak_factor, (self.num_slices, 10))
            demand[:, 0] *= 150 * peak_factor
            demand[:, 1] *= 0.3
            demand[:, 2] *= 15 * peak_factor
            demand[:, 3] *= 80 * peak_factor

        elif scenario == 'flash_crowd':
            # Sudden spike at 40-60% of simulation
            if 0.4 <= t <= 0.6:
                spike = 3.0
            else:
                spike = 1.0
            demand = np.random.exponential(0.5 * spike, (self.num_slices, 10))
            demand[:, 0] *= 120 * spike
            demand[:, 1] *= 0.4
            demand[:, 2] *= 12 * spike
            demand[:, 3] *= 60 * spike

        elif scenario == 'heterogeneous':
            # Different slice types have vastly different demands
            demand = np.random.exponential(0.5, (self.num_slices, 10))
            for i in range(self.num_slices):
                slice_type = i % 4
                if slice_type == 0:    # eMBB - high bandwidth
                    demand[i, 0] *= 200
                    demand[i, 1] *= 1.0
                elif slice_type == 1:  # uRLLC - low latency
                    demand[i, 0] *= 20
                    demand[i, 1] *= 0.01
                elif slice_type == 2:  # mMTC - many connections
                    demand[i, 0] *= 50
                    demand[i, 1] *= 5.0
                else:                  # XR - high bandwidth + low latency
                    demand[i, 0] *= 250
                    demand[i, 1] *= 0.1

        else:
            demand = np.random.exponential(0.5, (self.num_slices, 10))

        return demand.astype(np.float32)

    def compute_qos(
        self,
        allocation: np.ndarray,
        demand: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Compute QoS metrics given allocation and demand.
        Returns: (qos_satisfaction, energy_cost, latency, fairness)
        """
        demand_4d = demand[:, :4]

        # QoS satisfaction: how well demand is met
        satisfaction = np.minimum(allocation / (demand_4d + 1e-8), 1.0)
        qos = float(np.mean(satisfaction))

        # Energy cost: proportional to allocation with super-linear scaling
        energy = float(np.sum(allocation[:, 3] ** 1.2) / (self.total_resources[3] ** 1.2 + 1e-8))

        # Latency: inversely proportional to compute allocation
        latency = float(np.mean(demand_4d[:, 1] / (allocation[:, 1] + 1e-8)))
        latency = min(latency, 100.0)

        # Jain's fairness index
        s = satisfaction.mean(axis=1)
        fairness = float(np.sum(s) ** 2 / (self.num_slices * np.sum(s ** 2) + 1e-8))

        return qos, energy, latency, fairness

    def run_benchmark(
        self,
        algorithm_name: str,
        scenario: str = 'normal',
        num_episodes: int = 100,
        steps_per_episode: int = 200
    ) -> BenchmarkResult:
        """Run benchmark for a specific algorithm and scenario."""
        logger.info(f"Benchmarking {algorithm_name} on {scenario} scenario...")

        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        algorithm = self.algorithms[algorithm_name]
        episode_rewards = []
        episode_qos = []
        episode_energy = []
        episode_latency = []
        episode_fairness = []
        inference_times = []

        training_start = time.time()

        for ep in range(num_episodes):
            ep_reward = 0.0
            ep_qos_scores = []
            ep_energy_scores = []
            ep_latency_scores = []
            ep_fairness_scores = []

            for step in range(steps_per_episode):
                state = self.generate_scenario(scenario, step, steps_per_episode)

                # Measure inference time
                inf_start = time.time()
                allocation = algorithm.allocate(state.flatten(), step)
                inf_time = (time.time() - inf_start) * 1e6  # microseconds
                inference_times.append(inf_time)

                # Compute metrics
                qos, energy, latency, fairness = self.compute_qos(allocation, state)

                # Reward: QoS - energy penalty
                reward = qos - 0.3 * energy - 0.1 * min(latency, 10.0) / 10.0
                ep_reward += reward

                ep_qos_scores.append(qos)
                ep_energy_scores.append(energy)
                ep_latency_scores.append(latency)
                ep_fairness_scores.append(fairness)

            episode_rewards.append(ep_reward / steps_per_episode)
            episode_qos.append(np.mean(ep_qos_scores))
            episode_energy.append(np.mean(ep_energy_scores))
            episode_latency.append(np.mean(ep_latency_scores))
            episode_fairness.append(np.mean(ep_fairness_scores))

        training_time = time.time() - training_start

        # Find convergence (when variance stabilizes)
        convergence_ep = num_episodes
        for ep in range(50, num_episodes):
            recent_var = np.std(episode_rewards[ep-50:ep])
            if recent_var < 0.02 * abs(np.mean(episode_rewards[ep-50:ep]) + 1e-8):
                convergence_ep = ep
                break

        # SLA compliance (QoS > 0.9)
        sla_compliance = float(np.mean([1.0 if q > 0.9 else 0.0 for q in episode_qos]))

        return BenchmarkResult(
            algorithm=algorithm_name,
            scenario=scenario,
            episodes=num_episodes,
            avg_reward=float(np.mean(episode_rewards)),
            std_reward=float(np.std(episode_rewards)),
            avg_qos=float(np.mean(episode_qos)),
            avg_energy=float(np.mean(episode_energy)),
            avg_latency_ms=float(np.mean(episode_latency)),
            convergence_episode=convergence_ep,
            inference_time_us=float(np.mean(inference_times)),
            training_time_s=training_time,
            memory_mb=0.0,  # Would need psutil for actual measurement
            sla_compliance=sla_compliance,
            fairness_index=float(np.mean(episode_fairness)),
        )

    def run_all_benchmarks(
        self,
        scenarios: Optional[List[str]] = None,
        num_episodes: int = 100,
        steps_per_episode: int = 200
    ) -> List[BenchmarkResult]:
        """Run all algorithms across all scenarios."""
        if scenarios is None:
            scenarios = ['normal', 'peak_hour', 'flash_crowd', 'heterogeneous']

        results = []
        for scenario in scenarios:
            for algo_name in self.algorithms:
                try:
                    result = self.run_benchmark(
                        algo_name, scenario, num_episodes, steps_per_episode
                    )
                    results.append(result)
                    logger.info(
                        f"  {algo_name}/{scenario}: reward={result.avg_reward:.4f}, "
                        f"qos={result.avg_qos:.4f}, energy={result.avg_energy:.4f}"
                    )
                except Exception as e:
                    logger.error(f"  {algo_name}/{scenario} failed: {e}")

        return results

    def print_comparison_table(self, results: List[BenchmarkResult]):
        """Print a formatted comparison table."""
        print("\n" + "=" * 120)
        print(f"{'Algorithm':<20} {'Scenario':<15} {'Reward':>10} {'QoS':>8} "
              f"{'Energy':>8} {'Latency(ms)':>12} {'SLA%':>8} {'Fairness':>10} "
              f"{'Inf(μs)':>10}")
        print("-" * 120)
        for r in sorted(results, key=lambda x: (x.scenario, -x.avg_reward)):
            print(
                f"{r.algorithm:<20} {r.scenario:<15} {r.avg_reward:>10.4f} "
                f"{r.avg_qos:>8.4f} {r.avg_energy:>8.4f} {r.avg_latency_ms:>12.4f} "
                f"{r.sla_compliance:>7.1%} {r.fairness_index:>10.4f} "
                f"{r.inference_time_us:>10.2f}"
            )
        print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="DRL Benchmarking Tool")
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=200, help='Steps per episode')
    parser.add_argument('--slices', type=int, default=10, help='Number of slices')
    parser.add_argument('--scenario', type=str, default='normal',
                       choices=['normal', 'peak_hour', 'flash_crowd', 'heterogeneous'],
                       help='Traffic scenario')
    parser.add_argument('--algorithm', type=str, default=None,
                       help='Specific algorithm to benchmark')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all algorithms across all scenarios')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='results/benchmark_results.json',
                       help='Output file')

    args = parser.parse_args()

    benchmark = DRLBenchmark(num_slices=args.slices, seed=args.seed)

    if args.compare_all:
        results = benchmark.run_all_benchmarks(
            num_episodes=args.episodes,
            steps_per_episode=args.steps
        )
        benchmark.print_comparison_table(results)
    elif args.algorithm:
        result = benchmark.run_benchmark(
            args.algorithm, args.scenario, args.episodes, args.steps
        )
        print(f"\nResult: {asdict(result)}")
    else:
        # Default: run all algorithms on specified scenario
        results = []
        for algo in benchmark.algorithms:
            result = benchmark.run_benchmark(
                algo, args.scenario, args.episodes, args.steps
            )
            results.append(result)
        benchmark.print_comparison_table(results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(
            [asdict(r) for r in (results if 'results' in dir() else [result])],
            f, indent=2
        )
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
