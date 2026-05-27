import torch
import torch.optim as optim
import numpy as np
from data import sample_trajectory
from backbone import Backbone
from flow import VanillaFlow, flow_loss
from node import RoutingNode

# ── Stage 1 ──────────────────────────────────────────────────────────────────
backbone  = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
model     = VanillaFlow(backbone)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Stage 1: Training vanilla flow matching...")

for step in range(5000):
    x0, x1, xt, t, target_v = sample_trajectory(512)
    loss = flow_loss(model, x0, x1, xt, t, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"  Step {step} | Loss {loss.item():.4f}")

# Known analytically: var of mixture of N(-4,1) and N(4,1) = 16 + 1 = 17
sigma_mode = 17.0
print(f"\nStage 1 done. sigma_mode^2 = {sigma_mode}")

torch.save(model.state_dict(), "stage1_model.pt")
torch.save(backbone.state_dict(), "backbone.pt")  # keep this — stage 2 still needs it
print("Models saved.")

# Verify backbone immediately after saving
model.eval()
x_check = torch.randn(1000, 1)
with torch.no_grad():
    for i in range(200):
        t_val = i / 200
        t_c = torch.full((1000, 1), t_val)
        v_c = model(x_check, t_c)
        x_check = x_check + v_c / 200
c = x_check.numpy().flatten()
print(f"Backbone check: below -2: {(c<-2).sum()}, above +2: {(c>2).sum()}, middle: {((c>-2)&(c<2)).sum()}")
model.train()

# ── Stage 2 ──────────────────────────────────────────────────────────────────
node = RoutingNode(query_dim=32, v_dim=1)

# Orthogonal init — guaranteed balanced start, no randomness
# Measure actual query directions for each mode
backbone.eval()
with torch.no_grad():
    x_left  = torch.full((500, 1), -3.5)
    x_right = torch.full((500, 1),  3.5)
    t_late  = torch.full((500, 1),  0.9)
    q_left_mean  = backbone(x_left,  t_late).mean(0)
    q_right_mean = backbone(x_right, t_late).mean(0)
    # Normalize and set as keys
    node.k_left.data  = q_left_mean  / q_left_mean.norm()
    node.k_right.data = q_right_mean / q_right_mean.norm()
backbone.train()
print(f"Key similarity: {torch.nn.functional.cosine_similarity(node.k_left.unsqueeze(0), node.k_right.unsqueeze(0)).item():.4f}  (expected near -1.0)")

# Freeze backbone — stage 2 only trains the routing node
for param in backbone.parameters():
    param.requires_grad = False

optimizer2 = optim.Adam(node.parameters(), lr=1e-3)

N_warmup     = 500
lambda_final = 0.5
gamma_final  = 0.1

print("\nStage 2: Training routing node...")

for step in range(10000):
    warmup = min(1.0, step / N_warmup)
    lam    = lambda_final * warmup
    gam    = gamma_final  * warmup

    x0, x1, xt, t, target_v = sample_trajectory(256)

    q = backbone(xt, t)
    pred_v, scores, v_left, v_right = node(q)

    # Branch assignment by target sign
    left_mask  = (x1 < 0).squeeze()
    right_mask = (x1 > 0).squeeze()

    l_flow_left  = ((v_left[left_mask]   - target_v[left_mask])  ** 2).mean() if left_mask.sum()  > 0 else torch.tensor(0.0)
    l_flow_right = ((v_right[right_mask] - target_v[right_mask]) ** 2).mean() if right_mask.sum() > 0 else torch.tensor(0.0)
    l_flow = (l_flow_left + l_flow_right) / 2

    # Contrastive — push branch vector fields apart
    w_left  = node.proj_left.weight
    w_right = node.proj_right.weight
    l_contrast = torch.nn.functional.cosine_similarity(
        w_left.flatten().unsqueeze(0),
        w_right.flatten().unsqueeze(0)
    ).mean()

    # Entropy — stay uncertain early, allow confidence later
    entropy   = -(scores * (scores + 1e-8).log()).sum(-1).mean()
    l_entropy = -entropy

    # Query separation — left/right queries should point different directions
    if left_mask.sum() > 0 and right_mask.sum() > 0:
        q_left_mean  = q[left_mask].mean(0)
        q_right_mean = q[right_mask].mean(0)
        l_query_sep  = -torch.nn.functional.cosine_similarity(
            q_left_mean.unsqueeze(0),
            q_right_mean.unsqueeze(0)
        ).mean()
    else:
        l_query_sep = torch.tensor(0.0)

    # Balance — prevent one branch dominating
    mean_scores = scores.mean(0)
    l_balance   = ((mean_scores - 0.5) ** 2).sum()

    loss = l_flow + lam * l_contrast + gam * l_entropy + 1.0 * l_balance + 0.5 * l_query_sep

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    if step % 500 == 0:
        print(
            f"  Step {step:5d} | "
            f"Flow {l_flow.item():.4f} | "
            f"Contrast {l_contrast.item():.4f} | "
            f"Entropy {entropy.item():.4f} | "
            f"Balance {l_balance.item():.4f} | "
            f"QuerySep {l_query_sep.item():.4f}"
        )

torch.save(node.state_dict(), "node.pt")
print("\nNode saved.")