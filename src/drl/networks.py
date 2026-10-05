import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4, key_dim: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        
        self.query_proj = nn.Linear(input_dim, num_heads * key_dim)
        self.key_proj = nn.Linear(input_dim, num_heads * key_dim)
        self.value_proj = nn.Linear(input_dim, num_heads * key_dim)
        self.out_proj = nn.Linear(num_heads * key_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        q = self.query_proj(x).view(batch_size, self.num_heads, self.key_dim)
        k = self.key_proj(x).view(batch_size, self.num_heads, self.key_dim)
        v = self.value_proj(x).view(batch_size, self.num_heads, self.key_dim)
        
        # Attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) / (self.key_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(attn_weights, v)
        context = context.view(batch_size, self.num_heads * self.key_dim)
        
        out = self.out_proj(context)
        return out, attn_weights

class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 4, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        self.attention = AttentionModule(input_dim=state_dim, num_heads=4, key_dim=32)
        self.norm = nn.LayerNorm(state_dim)
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.hidden = nn.Sequential(*layers)
        
        # Mean output for continuous actions
        self.mu = nn.Linear(in_dim, action_dim)
        # Log std output for continuous actions (used in SAC)
        self.log_std = nn.Linear(in_dim, action_dim)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state):
        attn_out, attn_weights = self.attention(state)
        x = self.norm(state + attn_out) # Residual connection
        x = self.hidden(x)
        mu = torch.tanh(self.mu(x)) # Bounded between -1 and 1
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std, attn_weights

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 4, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        
        # Q1 architecture
        q1_layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            q1_layers.append(nn.Linear(in_dim, h_dim))
            q1_layers.append(nn.LayerNorm(h_dim))
            q1_layers.append(nn.ReLU())
            in_dim = h_dim
        q1_layers.append(nn.Linear(in_dim, 1))
        self.q1 = nn.Sequential(*q1_layers)
        
        # Q2 architecture
        q2_layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            q2_layers.append(nn.Linear(in_dim, h_dim))
            q2_layers.append(nn.LayerNorm(h_dim))
            q2_layers.append(nn.ReLU())
            in_dim = h_dim
        q2_layers.append(nn.Linear(in_dim, 1))
        self.q2 = nn.Sequential(*q2_layers)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        q1_val = self.q1(sa)
        q2_val = self.q2(sa)
        return q1_val, q2_val
