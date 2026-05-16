import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out

class ResNetPolicyValueNet(nn.Module):
    def __init__(self, num_colors=6, hidden_dim=192, num_res_blocks=8, num_actions=9, dropout=0.05, **kwargs):
        super().__init__()
        input_dim = 24 * num_colors 
        
        # Input Embedding
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # ResNet Towers
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])

        # Policy Head 
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        # Value Head 
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        
        x = self.input_fc(x)

        for block in self.res_blocks:
            x = block(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, x):
        #self.eval()
        #if isinstance(x, np.ndarray):
            #x = torch.from_numpy(x).float()

        if x.dim() == 2: 
            x = x.unsqueeze(0)
        
        #device = next(self.parameters()).device
        #x = x.to(device)

        policy_logits, value = self.forward(x)
        policy = F.softmax(policy_logits, dim=-1)

        return policy.squeeze(0), value.squeeze()
