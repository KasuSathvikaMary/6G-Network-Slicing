import os
import torch
import numpy as np
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Any
from .agent import MultiAgentOrchestrator
from .reward import RewardCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DRLTrainer:
    def __init__(self, orchestrator: MultiAgentOrchestrator, env: Any, config: Dict):
        self.orchestrator = orchestrator
        self.env = env
        self.config = config
        self.reward_calculator = RewardCalculator()
        
        self.epochs = config.get('epochs', 100)
        self.steps_per_epoch = config.get('steps_per_epoch', 1000)
        self.batch_size = config.get('batch_size', 256)
        self.eval_freq = config.get('eval_freq', 10)
        self.save_dir = config.get('save_dir', './models')
        self.warmup_steps = config.get('warmup_steps', 5000)
        
        # Setup schedulers for each agent
        self.schedulers = {}
        for st, agent in self.orchestrator.agents.items():
            self.schedulers[st] = {
                'actor': CosineAnnealingLR(agent.actor_optimizer, T_max=self.epochs * self.steps_per_epoch),
                'critic': CosineAnnealingLR(agent.critic_optimizer, T_max=self.epochs * self.steps_per_epoch)
            }
            
    def _step_schedulers(self):
        for st_scheds in self.schedulers.values():
            st_scheds['actor'].step()
            st_scheds['critic'].step()
            
    def train_online(self):
        logging.info("Starting online training...")
        total_steps = 0
        best_reward = -np.inf
        
        for epoch in range(self.epochs):
            states = self.env.reset()
            epoch_rewards = {st: 0.0 for st in self.orchestrator.slice_types}
            
            for step in range(self.steps_per_epoch):
                if total_steps < self.warmup_steps:
                    actions = {st: self.env.action_space.sample() for st in self.orchestrator.slice_types}
                else:
                    actions = self.orchestrator.get_actions(states, evaluate=False)
                    
                next_states, metrics_dict, requirements_dict, dones = self.env.step(actions)
                
                transitions = {}
                step_rewards = {}
                for st in self.orchestrator.slice_types:
                    reward = self.reward_calculator.compute_reward(metrics_dict[st], requirements_dict[st])
                    step_rewards[st] = reward
                    transitions[st] = (states[st], actions[st], reward, next_states[st], dones[st])
                    epoch_rewards[st] += reward
                    
                self.orchestrator.store_transitions(transitions)
                
                if total_steps >= self.batch_size:
                    losses = self.orchestrator.train_step(self.batch_size)
                    self._step_schedulers()
                    
                states = next_states
                total_steps += 1
                
                if all(dones.values()):
                    break
                    
            avg_reward = np.mean(list(epoch_rewards.values()))
            logging.info(f"Epoch {epoch+1}/{self.epochs}, Avg Reward: {avg_reward:.2f}")
            
            if (epoch + 1) % self.eval_freq == 0:
                eval_reward = self.evaluate()
                logging.info(f"Evaluation Reward: {eval_reward:.2f}")
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    logging.info("Saving best model...")
                    self.orchestrator.save_all(os.path.join(self.save_dir, "best_model"))
                    
            if (epoch + 1) % 50 == 0:
                self.orchestrator.save_all(os.path.join(self.save_dir, f"checkpoint_ep{epoch+1}"))
                
        logging.info("Training completed.")
        
    def evaluate(self, num_episodes: int = 5) -> float:
        total_rewards = []
        for _ in range(num_episodes):
            states = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                actions = self.orchestrator.get_actions(states, evaluate=True)
                next_states, metrics_dict, requirements_dict, dones = self.env.step(actions)
                for st in self.orchestrator.slice_types:
                    reward = self.reward_calculator.compute_reward(metrics_dict[st], requirements_dict[st])
                    episode_reward += reward
                states = next_states
                done = all(dones.values())
            total_rewards.append(episode_reward / len(self.orchestrator.slice_types))
        return np.mean(total_rewards)
        
    def train_offline(self, dataset):
        logging.info("Starting offline training from dataset...")
        # Populate buffers with dataset
        for transition in dataset:
            # Assume transition is formatted appropriately
            self.orchestrator.store_transitions(transition)
            
        for epoch in range(self.epochs):
            for step in range(self.steps_per_epoch):
                losses = self.orchestrator.train_step(self.batch_size)
                self._step_schedulers()
            
            if (epoch + 1) % self.eval_freq == 0:
                logging.info(f"Offline Training Epoch {epoch+1}/{self.epochs}")
                self.orchestrator.save_all(os.path.join(self.save_dir, f"offline_checkpoint_ep{epoch+1}"))
