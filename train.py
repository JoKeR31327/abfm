import torch
import torch.optim as optim
import numpy as np
from data import sample_trajectory
from backbone import Backbone
from flow import VanillaFlow, flow_loss
from node import RoutingNode

# ── Stage 1 ──────────────────────────────────────────────────────────────────
backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
model    = VanillaFlow(backbone)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

loss_history = []
window   = 50
epsilon  = 0.3
sigma_mode = None

print("Stage 1: Training vanilla flow matching...")

for step in range(10000):
    x0, x1, xt, t, target_v = sample_trajectory(256)

    loss = flow_loss(model, x0, x1, xt, t, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        loss_history.append(loss.item())

        if len(loss_history) > window:
            loss_history.pop(0)
            std = np.std(loss_history)
            print(f"Step {step} | Loss {loss.item():.4f} | Loss std {std:.4f}")

            if std < epsilon:
                # Estimate sigma_mode from x1 variance (data property, stable)
                with torch.no_grad():
                    _, x1_s, _, _, _ = sample_trajectory(1024)
                    sigma_mode = x1_s.var().item()
                print(f"\nStage 1 converged at step {step}.")
                print(f"sigma_mode^2 = {sigma_mode:.4f}")
                break

if sigma_mode is None:
    with torch.no_grad():
        _, x1_s, _, _, _ = sample_trajectory(1024)
        sigma_mode = x1_s.var().item()
    print(f"Stage 1 hit max steps. sigma_mode^2 = {sigma_mode:.4f}")


torch.save(backbone.state_dict(), "backbone.pt")
print("Backbone saved.")

# ── Stage 2 ──────────────────────────────────────────────────────────────────
node = RoutingNode(query_dim=32, v_dim=1)

# Orthogonal init — guaranteed balanced start
with torch.no_grad():
    node.k_left.data  = torch.zeros(32); node.k_left.data[0]  = 1.0
    node.k_right.data = torch.zeros(32); node.k_right.data[1] = 1.0

for param in backbone.parameters():
    param.requires_grad = False

optimizer2 = optim.Adam(
    node.parameters(),
    lr=1e-4
)

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
    entropy  = -(scores * (scores + 1e-8).log()).sum(-1).mean()
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

    if step % 200 == 0:
        print(
            f"Step {step} | "
            f"Flow {l_flow.item():.4f} | "
            f"Contrast {l_contrast.item():.4f} | "
            f"Entropy {entropy.item():.4f} | "
            f"Balance {l_balance.item():.4f} | "
            f"QuerySep {l_query_sep.item():.4f}"
        )

torch.save(backbone.state_dict(), "backbone.pt")
torch.save(node.state_dict(), "node.pt")
print("\nModels saved.")