import torch
import torch.nn as nn

class RoutingNode(nn.Module):
    def __init__(self, query_dim=32, v_dim=1):
        super().__init__()
        self.k_left  = nn.Parameter(torch.randn(query_dim))
        self.k_right = nn.Parameter(torch.randn(query_dim))
        self.proj_left  = nn.Linear(query_dim, v_dim)
        self.proj_right = nn.Linear(query_dim, v_dim)
    
    def forward(self, q):
        score_left  = (q * self.k_left).sum(-1)
        score_right = (q * self.k_right).sum(-1)
        scores = torch.softmax(
            torch.stack([score_left, score_right], dim=-1), dim=-1
        )
        v_left  = self.proj_left(q)
        v_right = self.proj_right(q)
        out = scores[:, 0:1] * v_left + scores[:, 1:2] * v_right
        return out, scores, v_left, v_right