import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .networks import ActorNetwork, CriticNetwork

class TransferLearningManager:
    def __init__(self, source_actor: ActorNetwork, source_critic: CriticNetwork, device: str = "cpu"):
        self.source_actor = source_actor
        self.source_critic = source_critic
        self.device = torch.device(device)
        
    def create_student_model(self, freeze_layers: bool = True) -> tuple:
        """Create a student model initialized with source model weights."""
        target_actor = copy.deepcopy(self.source_actor)
        target_critic = copy.deepcopy(self.source_critic)
        
        if freeze_layers:
            # Freeze all but the last layer (mu and log_std) in actor
            for name, param in target_actor.named_parameters():
                if 'mu' not in name and 'log_std' not in name:
                    param.requires_grad = False
                    
            # Freeze all but the last layer in critic
            for name, param in target_critic.named_parameters():
                if 'q1.6' not in name and 'q2.6' not in name:  # Assuming last layer is index 6
                    param.requires_grad = False
                    
        return target_actor.to(self.device), target_critic.to(self.device)
        
    def unfreeze_progressively(self, model: nn.Module, layer_idx: int):
        """Unfreeze specific layers for fine-tuning."""
        # Simple implementation to unfreeze layers progressively
        for param in model.parameters():
            param.requires_grad = True
            
    def compute_knowledge_distillation_loss(self, student_actor, states, temperature: float = 2.0):
        """Compute KD loss between source (teacher) and student actors."""
        with torch.no_grad():
            teacher_mu, teacher_log_std, _ = self.source_actor(states)
            teacher_std = teacher_log_std.exp()
            teacher_dist = torch.distributions.Normal(teacher_mu, teacher_std)
            teacher_action = teacher_dist.rsample()
            
        student_mu, student_log_std, _ = student_actor(states)
        student_std = student_log_std.exp()
        student_dist = torch.distributions.Normal(student_mu, student_std)
        
        # KL Divergence between teacher and student policies
        kl_div = torch.distributions.kl_divergence(teacher_dist, student_dist).mean()
        return kl_div
        
    def align_features(self, source_features, target_features):
        """Domain adaptation using Maximum Mean Discrepancy (MMD)."""
        # Simplified MMD calculation
        delta = source_features.mean(0) - target_features.mean(0)
        loss = torch.sum(delta * delta)
        return loss
        
    def compute_similarity(self, source_demands, target_demands) -> float:
        """Compute cosine similarity between slice type demands to determine transfer feasibility."""
        source_vec = torch.FloatTensor(source_demands).view(-1)
        target_vec = torch.FloatTensor(target_demands).view(-1)
        sim = F.cosine_similarity(source_vec, target_vec, dim=0)
        return sim.item()
