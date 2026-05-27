import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, xt, t):
        # xt: [B, 1], t: [B, 1]
        inp = torch.cat([xt, t], dim=-1)  # [B, 2]
        return self.net(inp)              # [B, 32]