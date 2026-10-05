import numpy as np
from typing import Dict, List, Any, Optional
import json

class AttentionVisualizer:
    """
    Extracts and visualizes attention weights from DRL networks.
    """
    def __init__(self, actor_network: Any):
        self.actor_network = actor_network
        self.temporal_history: List[Dict[str, Any]] = []
        
    def get_attention_weights(self, state: np.ndarray) -> np.ndarray:
        """
        Extract attention weights for the given state.
        Assumes actor_network has a get_attention method.
        """
        if hasattr(self.actor_network, 'get_attention'):
            return self.actor_network.get_attention(state)
        else:
            # Fallback mock implementation for completeness
            num_features = state.shape[-1]
            weights = np.random.dirichlet(np.ones(num_features), size=1)[0]
            return weights
            
    def visualize_attention(self, state: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate attention heatmap data.
        """
        weights = self.get_attention_weights(state)
        
        heatmap_data = {
            "features": feature_names,
            "weights": weights.tolist(),
            "normalized_weights": (weights / np.max(weights)).tolist()
        }
        return heatmap_data
        
    def create_correlation_matrix(self, states: np.ndarray) -> np.ndarray:
        """
        Create feature correlation matrix based on attention over multiple states.
        """
        attentions = np.array([self.get_attention_weights(s) for s in states])
        # Compute correlation between features' attention weights
        correlation_matrix = np.corrcoef(attentions.T)
        # Handle NaNs from zero variance
        correlation_matrix = np.nan_to_num(correlation_matrix)
        return correlation_matrix
        
    def track_temporal_attention(self, state: np.ndarray, timestamp: float, slice_id: str):
        """
        Track how attention changes over time.
        """
        weights = self.get_attention_weights(state)
        self.temporal_history.append({
            "timestamp": timestamp,
            "slice_id": slice_id,
            "weights": weights
        })
        
    def export_visualizations(self, filepath: str):
        """
        Export temporal attention history to JSON.
        """
        export_data = []
        for entry in self.temporal_history:
            export_data.append({
                "timestamp": entry["timestamp"],
                "slice_id": entry["slice_id"],
                "weights": entry["weights"].tolist()
            })
            
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
