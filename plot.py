import torch
import matplotlib.pyplot as plt
from backbone import Backbone
from flow import VanillaFlow
from node import RoutingNode

# Load models
backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
backbone.load_state_dict(torch.load("backbone.pt", weights_only=True))
backbone.eval()

model = VanillaFlow(backbone)
model.load_state_dict(torch.load("stage1_model.pt", weights_only=True))
model.eval()

node = RoutingNode(query_dim=32, v_dim=1)
node.load_state_dict(torch.load("node.pt", weights_only=True))
node.eval()

# Stage 1 check — vanilla flow only
x = torch.randn(1000, 1)
with torch.no_grad():
    for i in range(200):
        t = torch.full((1000, 1), i / 200)
        x = x + model(x, t) / 200
s = x.numpy().flatten()
print(f"Stage1: below -2: {(s<-2).sum()}, above +2: {(s>2).sum()}, middle: {((s>-2)&(s<2)).sum()}")

# Gate 1 — routing node inference
x = torch.randn(1000, 1)
all_scores = []
with torch.no_grad():
    for i in range(200):
        t = torch.full((1000, 1), i / 200)
        q = backbone(x, t)
        v, scores, _, _ = node(q)
        all_scores.append(scores[:, 0].mean().item())
        x = x + v / 200
s = x.numpy().flatten()
print(f"Gate1:  below -2: {(s<-2).sum()}, above +2: {(s>2).sum()}, middle: {((s>-2)&(s<2)).sum()}")
print(f"Score at t=0: {all_scores[0]:.4f}, t=0.5: {all_scores[100]:.4f}, t=1.0: {all_scores[199]:.4f}")

# Query similarity
x_left  = torch.full((100, 1), -3.5)
x_right = torch.full((100, 1),  3.5)
t_late  = torch.full((100, 1),  0.9)
with torch.no_grad():
    sim = torch.nn.functional.cosine_similarity(
        backbone(x_left, t_late).mean(0, keepdim=True),
        backbone(x_right, t_late).mean(0, keepdim=True)
    )
print(f"Query similarity at t=0.9: {sim.item():.4f}")