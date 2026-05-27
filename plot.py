import torch
import matplotlib.pyplot as plt
from backbone import Backbone
from flow import VanillaFlow
from node import RoutingNode



# Load models
backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)

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


# Per-sample score trajectory check
torch.manual_seed(42)

# Pick one sample heading left, one heading right
x_left_sample  = torch.tensor([[-3.5]])
x_right_sample = torch.tensor([[3.5]])

steps = 200
scores_left_traj  = []
scores_right_traj = []

xt_left  = x_left_sample.clone()
xt_right = x_right_sample.clone()

with torch.no_grad():
    for i in range(steps):
        t_val = i / steps
        t = torch.full((1, 1), t_val)

        q_l = backbone(xt_left,  t)
        q_r = backbone(xt_right, t)

        _, scores_l, _, _ = node(q_l)
        _, scores_r, _, _ = node(q_r)

        scores_left_traj.append(scores_l[0, 0].item())   # left branch score for left sample
        scores_right_traj.append(scores_r[0, 1].item())  # right branch score for right sample

        xt_left  = xt_left  + node(q_l)[0] / steps
        xt_right = xt_right + node(q_r)[0] / steps

print(f"\nPer-sample score trajectories:")
print(f"Left  sample — score_left:  t=0: {scores_left_traj[0]:.4f}  t=0.5: {scores_left_traj[100]:.4f}  t=1.0: {scores_left_traj[199]:.4f}")
print(f"Right sample — score_right: t=0: {scores_right_traj[0]:.4f}  t=0.5: {scores_right_traj[100]:.4f}  t=1.0: {scores_right_traj[199]:.4f}")


