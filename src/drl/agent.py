import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
from typing import Dict, List, Tuple
from .networks import ActorNetwork, CriticNetwork
from .replay_buffer import PrioritizedReplayBuffer

class SliceAgent:
    def __init__(self, agent_id: str, state_dim: int = 10, action_dim: int = 4, 
                 lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005, 
                 alpha: float = 0.2, buffer_capacity: int = 100000, device="cpu"):
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.target_entropy = -torch.prod(torch.Tensor([action_dim]).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha
        
        self.memory = PrioritizedReplayBuffer(buffer_capacity)
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, log_std, _ = self.actor(state)
            if evaluate:
                action = mu
            else:
                std = log_std.exp()
                dist = Normal(mu, std)
                z = dist.sample()
                action = torch.tanh(z)
        return action.cpu().numpy()[0]
        
    def add_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done, self.agent_id)
        
    def _soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def update(self, batch_size: int = 256):
        if self.memory.tree.n_entries < batch_size:
            return 0.0, 0.0
            
        states, actions, rewards, next_states, dones, _, idxs, is_weights = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            next_mu, next_log_std, _ = self.actor(next_states)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mu, next_std)
            next_z = next_dist.sample()
            next_action = torch.tanh(next_z)
            next_log_pi = next_dist.log_prob(next_z) - torch.log(1 - next_action.pow(2) + 1e-6)
            next_log_pi = next_log_pi.sum(1, keepdim=True)
            
            q1_next, q2_next = self.critic_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + (1 - dones) * self.gamma * q_next
            
        q1, q2 = self.critic(states, actions)
        critic_loss1 = F.mse_loss(q1, q_target, reduction='none')
        critic_loss2 = F.mse_loss(q2, q_target, reduction='none')
        critic_loss = (critic_loss1 * weights).mean() + (critic_loss2 * weights).mean()
        
        # Priority update
        td_errors = (torch.min(q1, q2) - q_target).detach().cpu().numpy()
        self.memory.update_priorities(idxs, td_errors)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        mu, log_std, _ = self.actor(states)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.sample()
        action_pi = torch.tanh(z)
        log_pi = dist.log_prob(z) - torch.log(1 - action_pi.pow(2) + 1e-6)
        log_pi = log_pi.sum(1, keepdim=True)
        
        q1_pi, q2_pi = self.critic(states, action_pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
        
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{self.agent_id}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{self.agent_id}_critic.pth"))
        
    def load(self, directory: str):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{self.agent_id}_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{self.agent_id}_critic.pth")))
        self.critic_target.load_state_dict(self.critic.state_dict())

class MultiAgentOrchestrator:
    def __init__(self, slice_types: List[str], state_dim: int = 10, action_dim: int = 4, device="cpu"):
        self.agents = {st: SliceAgent(st, state_dim, action_dim, device=device) for st in slice_types}
        self.slice_types = slice_types
        
    def get_actions(self, states: Dict[str, np.ndarray], evaluate: bool = False) -> Dict[str, np.ndarray]:
        actions = {}
        for st, state in states.items():
            if st in self.agents:
                actions[st] = self.agents[st].select_action(state, evaluate)
        return actions
        
    def store_transitions(self, transitions: Dict[str, Tuple]):
        for st, (state, action, reward, next_state, done) in transitions.items():
            if st in self.agents:
                self.agents[st].add_experience(state, action, reward, next_state, done)
                
    def train_step(self, batch_size: int = 256) -> Dict[str, Dict[str, float]]:
        losses = {}
        for st, agent in self.agents.items():
            a_loss, c_loss = agent.update(batch_size)
            losses[st] = {'actor_loss': a_loss, 'critic_loss': c_loss}
        return losses
        
    def save_all(self, directory: str):
        for agent in self.agents.values():
            agent.save(directory)
            
    def load_all(self, directory: str):
        for agent in self.agents.values():
            agent.load(directory)
