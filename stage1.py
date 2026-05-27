import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from data import sample_trajectory
from backbone import Backbone
from flow import VanillaFlow, flow_loss

backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
model = VanillaFlow(backbone)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for step in range(5000):
    x0, x1, xt, t, target_v = sample_trajectory(512)
    loss = flow_loss(model, x0, x1, xt, t, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 500 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

# Generate samples
model.eval()
x = torch.randn(1000, 1)
steps = 200
with torch.no_grad():
    for i in range(steps):
        t_val = i / steps
        t = torch.full((1000, 1), t_val)
        v = model(x, t)
        x = x + v / steps

samples = x.numpy().flatten()
print(f"\nMean: {samples.mean():.3f}  Std: {samples.std():.3f}")
print(f"Below -2: {(samples < -2).sum()}  Above +2: {(samples > 2).sum()}  Middle: {((samples > -2) & (samples < 2)).sum()}")

plt.hist(samples, bins=50)
plt.title("Stage 1 Only — Should show two peaks at -4 and +4")
plt.savefig("test_stage1.png")
print("Saved test_stage1.png")