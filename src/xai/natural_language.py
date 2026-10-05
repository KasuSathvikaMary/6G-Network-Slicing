from typing import Dict, Any, List

class NLExplainer:
    """
    Template-based natural language explanation generation.
    """
    def __init__(self):
        self.verbosity = 'detailed'
        self.templates = {
            'allocation_increase': {
                'brief': "Increased {resource} for {slice_id} to handle load.",
                'detailed': "Increased {resource} allocation for slice {slice_id} from {old_val} to {new_val} due to high utilization.",
                'technical': "Action: Increase. Resource: {resource}, Target: {slice_id}, Diff: +{diff}, Trigger: Utilization > Threshold."
            },
            'allocation_decrease': {
                'brief': "Decreased {resource} for {slice_id} to save energy.",
                'detailed': "Decreased {resource} allocation for slice {slice_id} from {old_val} to {new_val} due to low utilization, saving energy.",
                'technical': "Action: Decrease. Resource: {resource}, Target: {slice_id}, Diff: -{diff}, Trigger: Utilization < Threshold."
            },
            'slice_creation': {
                'brief': "Created new slice {slice_id}.",
                'detailed': "Created new slice {slice_id} with initial profile {profile}.",
                'technical': "Action: Create. Target: {slice_id}, Profile: {profile}."
            },
            'sla_violation_response': {
                'brief': "Adjusted resources for {slice_id} due to SLA violation.",
                'detailed': "Increased resources for slice {slice_id} to mitigate detected SLA violation (Latency > {max_latency}ms).",
                'technical': "Action: Mitigate_SLA. Target: {slice_id}, Metric: Latency, Threshold: {max_latency}ms, Priority: Critical."
            },
            'energy_optimization': {
                'brief': "Optimized energy across slices.",
                'detailed': "Applied power capping to non-critical slices to reduce total energy consumption by {savings}%.",
                'technical': "Action: Energy_Cap. Strategy: Non-Critical_Throttling, Projected_Savings: {savings}%, Constraints: Maintain QoS."
            }
        }
        
    def set_verbosity(self, level: str):
        """
        Set the verbosity level: 'brief', 'detailed', or 'technical'.
        """
        if level not in ['brief', 'detailed', 'technical']:
            raise ValueError(f"Unknown verbosity level: {level}")
        self.verbosity = level
        
    def get_explanation_template(self, decision_type: str) -> str:
        """
        Get the template for a specific decision type and current verbosity.
        """
        if decision_type not in self.templates:
            return "Decision made for {slice_id}."
        return self.templates[decision_type][self.verbosity]
        
    def generate_explanation(self, decision_type: str, context: Dict[str, Any]) -> str:
        """
        Generate a natural language explanation based on decision type and context.
        """
        template = self.get_explanation_template(decision_type)
        
        # Calculate diffs if needed
        if 'old_val' in context and 'new_val' in context and 'diff' not in context:
            context['diff'] = abs(context['new_val'] - context['old_val'])
            
        try:
            explanation = template.format(**context)
        except KeyError as e:
            explanation = f"Missing context parameter for explanation: {e}"
            
        return explanation
