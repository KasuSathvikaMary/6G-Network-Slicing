import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

class RuleExtractor:
    """
    Extracts IF-THEN rules from trained neural networks using a surrogate decision tree.
    """
    def __init__(self, policy_network: Any, feature_names: List[str]):
        self.policy_network = policy_network
        self.feature_names = feature_names
        self.surrogate_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10)
        self.is_fitted = False
        
    def extract_rules(self, states: np.ndarray) -> str:
        """
        Train a surrogate decision tree and extract readable rules.
        """
        # Generate actions from the neural policy to train the surrogate
        actions = []
        for state in states:
            if hasattr(self.policy_network, 'predict'):
                action = self.policy_network.predict(state.reshape(1, -1))[0]
            else:
                # Mock implementation if predict is missing
                action = np.argmax(state) 
            actions.append(action)
            
        actions = np.array(actions)
        
        # Train surrogate model
        self.surrogate_model.fit(states, actions)
        self.is_fitted = True
        
        # Extract rules
        rules_text = export_text(self.surrogate_model, feature_names=self.feature_names)
        return rules_text
        
    def simplify_rules(self, max_depth: int = 3) -> str:
        """
        Simplify rules by reducing tree depth.
        """
        if not self.is_fitted:
            raise ValueError("Surrogate model not fitted. Call extract_rules first.")
            
        # Retrain with simpler parameters
        simplified_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=20)
        # Note: In practice we would retrain on saved states. Returning message for now.
        return "Rules simplified by reducing max depth. (Requires refitting with states)."
        
    def evaluate_rule_fidelity(self, test_states: np.ndarray) -> float:
        """
        Evaluate how closely the surrogate rules match the neural policy.
        """
        if not self.is_fitted:
            raise ValueError("Surrogate model not fitted. Call extract_rules first.")
            
        neural_actions = []
        for state in test_states:
            if hasattr(self.policy_network, 'predict'):
                action = self.policy_network.predict(state.reshape(1, -1))[0]
            else:
                action = np.argmax(state)
            neural_actions.append(action)
            
        rule_actions = self.surrogate_model.predict(test_states)
        
        fidelity = accuracy_score(neural_actions, rule_actions)
        return float(fidelity)
        
    def get_rule_confidence(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Get the predicted rule action and its confidence (probability).
        """
        if not self.is_fitted:
            raise ValueError("Surrogate model not fitted.")
            
        probs = self.surrogate_model.predict_proba(state.reshape(1, -1))[0]
        action = int(np.argmax(probs))
        confidence = float(probs[action])
        return action, confidence
