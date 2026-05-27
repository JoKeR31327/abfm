import torch
import torch.optim as optim
import torch.nn.functional as F
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

print("\nStage 1 done. sigma_mode^2 = 17.0")

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

print("\nStage 2: Training routing node...")

for step in range(10000):
    warmup = min(1.0, step / N_warmup)
    lam    = lambda_final * warmup

    x0, x1, xt, t, target_v = sample_trajectory(256)

    q = backbone(xt, t)
    _, scores, v_left, v_right = node(q)

    # Unsupervised routed flow loss: blend experts by the soft routing scores
    # and compute MSE against the target velocity. This removes explicit
    # supervision on routing (no CE, no masked branch losses).
    routing = scores  # shape [B, 2]
    v_routed = routing[:, 0:1] * v_left + routing[:, 1:2] * v_right
    l_flow = ((v_routed - target_v) ** 2).mean()

    # Contrastive — push branch vector fields apart
    w_left  = node.proj_left.weight
    w_right = node.proj_right.weight
    l_contrast = torch.nn.functional.cosine_similarity(
        w_left.flatten().unsqueeze(0),
        w_right.flatten().unsqueeze(0)
    ).mean()

    loss = l_flow + lam * l_contrast

    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    if step % 500 == 0:
        print(
            f"  Step {step:5d} | "
            f"Flow {l_flow.item():.4f} | "
            f"Contrast {l_contrast.item():.4f}"
        )

torch.save(node.state_dict(), "node.pt")
print("\nNode saved.")