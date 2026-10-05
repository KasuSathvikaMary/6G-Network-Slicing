import numpy as np
from typing import List, Tuple, Dict, Any

class ParetoAnalyzer:
    """
    Computes Pareto-optimal solutions for multi-objective optimization (e.g., NSGA-II style).
    """
    def __init__(self):
        pass

    def is_dominated(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """
        Check if obj1 is dominated by obj2.
        Assuming minimization for all objectives.
        """
        return np.all(obj2 <= obj1) and np.any(obj2 < obj1)

    def compute_pareto_front(self, population_objs: np.ndarray) -> List[int]:
        """
        Perform non-dominated sorting to find the Pareto front (Rank 1).
        population_objs: 2D array of shape (n_individuals, n_objectives)
        Returns list of indices of the non-dominated individuals.
        """
        n = population_objs.shape[0]
        pareto_front = []
        
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j and self.is_dominated(population_objs[i], population_objs[j]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(i)
                
        return pareto_front

    def calculate_crowding_distance(self, population_objs: np.ndarray, front_indices: List[int]) -> np.ndarray:
        """
        Calculate crowding distance for individuals in a specific front.
        """
        num_individuals = len(front_indices)
        num_objectives = population_objs.shape[1]
        distances = np.zeros(num_individuals)
        
        if num_individuals <= 2:
            distances[:] = np.inf
            return distances
            
        for m in range(num_objectives):
            sorted_indices = np.argsort(population_objs[front_indices, m])
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            
            obj_min = population_objs[front_indices[sorted_indices[0]], m]
            obj_max = population_objs[front_indices[sorted_indices[-1]], m]
            
            if obj_max == obj_min:
                continue
                
            for i in range(1, num_individuals - 1):
                distances[sorted_indices[i]] += (
                    population_objs[front_indices[sorted_indices[i+1]], m] - 
                    population_objs[front_indices[sorted_indices[i-1]], m]
                ) / (obj_max - obj_min)
                
        return distances

    def find_knee_point(self, population_objs: np.ndarray, front_indices: List[int]) -> int:
        """
        Find the 'knee point' on the Pareto front (maximum marginal return).
        Uses a simple distance to utopian point metric.
        """
        if not front_indices:
            raise ValueError("Front indices cannot be empty")
            
        front_objs = population_objs[front_indices]
        # Utopian point (ideal point)
        utopian_point = np.min(front_objs, axis=0)
        
        # Normalize objectives
        max_objs = np.max(front_objs, axis=0)
        min_objs = np.min(front_objs, axis=0)
        range_objs = max_objs - min_objs
        range_objs[range_objs == 0] = 1.0 # Avoid division by zero
        
        normalized_objs = (front_objs - utopian_point) / range_objs
        
        # Distance to utopian point (origin in normalized space)
        distances = np.linalg.norm(normalized_objs, axis=1)
        
        knee_idx = np.argmin(distances)
        return front_indices[knee_idx]
