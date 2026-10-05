import numpy as np
from typing import List, Dict, Tuple

class DVFSController:
    """
    Dynamic Voltage and Frequency Scaling controller.
    """
    def __init__(self):
        # Example P-states: (Frequency in GHz, Voltage in V)
        self.p_states = [
            (3.0, 1.2),
            (2.5, 1.1),
            (2.0, 1.0),
            (1.5, 0.9),
            (1.0, 0.8)
        ]
        self.capacitance = 1.0 # Effective capacitance constant
        self.transition_latency_ms = 10.0 # Time to switch states
        self.current_state_idx = 0
        
    def get_power(self, frequency: float, voltage: float) -> float:
        """
        P = C * V^2 * f
        """
        return self.capacitance * (voltage ** 2) * frequency
        
    def get_optimal_frequency(self, target_load: float) -> Tuple[float, float, int]:
        """
        Select lowest frequency that can handle the target load.
        target_load: Required compute capacity in GHz.
        Returns: (Frequency, Voltage, StateIndex)
        """
        optimal_idx = 0 # Default to max performance
        
        # Sort states by frequency ascending
        sorted_states = sorted(enumerate(self.p_states), key=lambda x: x[1][0])
        
        for idx, (freq, vol) in sorted_states:
            if freq >= target_load:
                return freq, vol, idx
                
        # If load exceeds max frequency, return max
        return self.p_states[0][0], self.p_states[0][1], 0
        
    def set_frequency(self, slice_id: str, target_load: float) -> Dict[str, float]:
        """
        Set frequency for a slice based on its load.
        """
        freq, vol, idx = self.get_optimal_frequency(target_load)
        latency = 0.0
        if idx != self.current_state_idx:
            latency = self.transition_latency_ms
            self.current_state_idx = idx
            
        power = self.get_power(freq, vol)
        
        return {
            "slice_id": slice_id,
            "frequency_ghz": freq,
            "voltage_v": vol,
            "power_w": power,
            "transition_latency_ms": latency
        }

    def get_power_savings(self, target_load: float) -> float:
        """
        Calculate power saved by using optimal frequency vs max frequency.
        """
        max_f, max_v = self.p_states[0]
        max_power = self.get_power(max_f, max_v)
        
        opt_f, opt_v, _ = self.get_optimal_frequency(target_load)
        opt_power = self.get_power(opt_f, opt_v)
        
        return max_power - opt_power
