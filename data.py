# data.py
import torch
import matplotlib.pyplot as plt

def sample_target(n):
    half = n // 2
    left  = torch.randn(half, 1) - 4.0
    right = torch.randn(n - half, 1) + 4.0
    return torch.cat([left, right], dim=0)

def sample_source(n):
    return torch.randn(n, 1)

def sample_trajectory(n):
    x0 = sample_source(n)
    x1 = sample_target(n)
    t  = torch.rand(n, 1)
    xt = t * x1 + (1 - t) * x0
    target_v = x1 - x0
    return x0, x1, xt, t, target_v

if __name__ == "__main__":
    x1 = sample_target(1000)
    plt.hist(x1.numpy().flatten(), bins=50)
    plt.title("Target Distribution — Two Gaussians")
    plt.xlabel("x")
    plt.savefig("target_distribution.png")
    print("Saved target_distribution.png")
    print(f"Mean: {x1.mean():.3f}  Std: {x1.std():.3f}")
    print(f"Shape: {x1.shape}")