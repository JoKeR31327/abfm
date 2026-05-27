import torch
import torch.nn as nn

class VanillaFlow(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Simple head: query space to velocity space
        self.head = nn.Linear(32, 1)
    
    def forward(self, xt, t):
        q = self.backbone(xt, t)
        return self.head(q)  # predicted velocity

def flow_loss(model, x0, x1, xt, t, target_v):
    pred_v = model(xt, t)
    return ((pred_v - target_v) ** 2).mean()