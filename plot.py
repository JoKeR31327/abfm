import torch
import matplotlib.pyplot as plt

def plot_stage1(model, n=500):
    x0 = torch.randn(n, 1)
    xt = x0.clone()
    
    # Euler integration
    steps = 100
    for i in range(steps):
        t_val = i / steps
        t = torch.full((n, 1), t_val)
        with torch.no_grad():
            v = model(xt, t)
        xt = xt + v * (1.0 / steps)
    
    plt.figure(figsize=(8, 4))
    plt.hist(xt.numpy().flatten(), bins=50)
    plt.title("Stage 1 Generated Samples")
    plt.xlabel("x")
    plt.savefig("stage1_samples.png")
    plt.close()
    print("Saved stage1_samples.png")



def plot_gate1(backbone, node, n=500):
    x0 = torch.randn(n, 1)
    xt = x0.clone()
    all_scores = []
    
    steps = 100
    for i in range(steps):
        t_val = i / steps
        t = torch.full((n, 1), t_val)
        with torch.no_grad():
            q = backbone(xt, t)
            v, scores, _, _ = node(q)
            all_scores.append(scores[:, 0].mean().item())
        xt = xt + v * (1.0 / steps)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Generated samples
    ax1.hist(xt.numpy().flatten(), bins=50)
    ax1.set_title("Gate 1: Generated Samples")
    ax1.set_xlabel("x")
    
    # Attention scores over t
    ax2.plot(all_scores)
    ax2.axhline(0.5, color='r', linestyle='--', label='0.5')
    ax2.set_title("Attention Score (left branch) over t")
    ax2.set_xlabel("timestep")
    ax2.set_ylabel("score")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("gate1_result.png")
    plt.close()
    print("Saved gate1_result.png")



import torch
from backbone import Backbone
from flow import VanillaFlow
from node import RoutingNode

backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
backbone.load_state_dict(torch.load("backbone.pt", weights_only=True))

model = VanillaFlow(backbone)

node = RoutingNode(query_dim=32, v_dim=1)
node.load_state_dict(torch.load("node.pt", weights_only=True))

backbone.eval()
node.eval()

plot_stage1(model)
plot_gate1(backbone, node)


if __name__ == "__main__":
    import torch
    
    # Create a test node
    node = RoutingNode(query_dim=32, v_dim=1)
    
    # Simulate early trajectory — query near zero (Gaussian noise)
    print("=== Early trajectory (t near 0) ===")
    print("Expected: scores near [0.5, 0.5] — uncertain routing")
    q_early = torch.randn(5, 32) * 0.1  # small query — near noise
    out, scores, v_left, v_right = node(q_early)
    print(f"Scores:   {scores.mean(0).tolist()}")
    print(f"Output v: {out.mean().item():.4f}")
    print()
    
    # Simulate late trajectory — query strongly positive (heading to right mode)
    print("=== Late trajectory, right-going sample (t near 1) ===")
    print("Expected: scores near [0.0, 1.0] — confident right branch")
    q_right = torch.ones(5, 32) * 2.0  # strong positive query
    out, scores, v_left, v_right = node(q_right)
    print(f"Scores:   {scores.mean(0).tolist()}")
    print(f"Output v: {out.mean().item():.4f} (expected near +8.0)")
    print()
    
    # Simulate late trajectory — query strongly negative (heading to left mode)
    print("=== Late trajectory, left-going sample (t near 1) ===")
    print("Expected: scores near [1.0, 0.0] — confident left branch")
    q_left = torch.ones(5, 32) * -2.0  # strong negative query
    out, scores, v_left, v_right = node(q_left)
    print(f"Scores:   {scores.mean(0).tolist()}")
    print(f"Output v: {out.mean().item():.4f} (expected near -8.0)")



# Print diagnostic results
print("\n=== GATE 1 DIAGNOSTIC ===")

# Re-run inference and collect data
x0 = torch.randn(1000, 1)
xt = x0.clone()
all_scores = []

steps = 100
for i in range(steps):
    t_val = i / steps
    t = torch.full((1000, 1), t_val)
    with torch.no_grad():
        q = backbone(xt, t)
        v, scores, _, _ = node(q)
        all_scores.append(scores[:, 0].mean().item())
    xt = xt + v * (1.0 / steps)

final_samples = xt.numpy().flatten()

print(f"\nGenerated samples stats:")
print(f"  Mean:  {final_samples.mean():.4f}  (expected near 0.0 if both modes captured)")
print(f"  Std:   {final_samples.std():.4f}   (expected ~4.0 for two modes at -4 and +4)")
print(f"  Min:   {final_samples.min():.4f}")
print(f"  Max:   {final_samples.max():.4f}")
print(f"  Samples below -2: {(final_samples < -2).sum()} (expected ~500)")
print(f"  Samples above +2: {(final_samples > 2).sum()} (expected ~500)")
print(f"  Samples between -2 and +2: {((final_samples > -2) & (final_samples < 2)).sum()} (expected ~0)")

print(f"\nAttention score over trajectory:")
print(f"  Score at t=0:   {all_scores[0]:.4f}  (expected near 0.5)")
print(f"  Score at t=0.5: {all_scores[50]:.4f} (expected moving away from 0.5)")
print(f"  Score at t=1.0: {all_scores[99]:.4f} (expected near 0.0 or 1.0)")
print(f"  Score range: [{min(all_scores):.4f}, {max(all_scores):.4f}]")

print("\n=== VERDICT ===")
left_count = (final_samples < -2).sum()
right_count = (final_samples > 2).sum()
middle_count = ((final_samples > -2) & (final_samples < 2)).sum()
score_range = max(all_scores) - min(all_scores)

if left_count > 300 and right_count > 300 and middle_count < 100:
    print("PASS — Two clean modes detected. Gate 1 succeeded.")
elif score_range > 0.3:
    print("PARTIAL — Routing is diverging but sample separation incomplete. Needs more training.")
else:
    print("FAIL — Samples not separating. Routing not learning. Diagnose further.")


# add to plot.py temporarily
model.eval()
x = torch.randn(1000, 1)
steps = 100
with torch.no_grad():
    for i in range(steps):
        t_val = i / steps
        t = torch.full((1000, 1), t_val)
        v = model(x, t)
        x = x + v / steps
samples = x.numpy().flatten()
print(f"Stage1 model: below -2: {(samples<-2).sum()}, above +2: {(samples>2).sum()}, middle: {((samples>-2)&(samples<2)).sum()}")