import torch
import torch.nn.functional as F
from backbone import Backbone
from node import RoutingNode

print("=" * 50)
print("GATE 1 COMPLETENESS TEST")
print("=" * 50)

# Load trained models
backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
backbone.load_state_dict(torch.load("backbone.pt", weights_only=True))
backbone.eval()

node = RoutingNode(query_dim=32, v_dim=1)
node.load_state_dict(torch.load("node.pt", weights_only=True))
node.eval()

# ── TEST 1: Sample separation ──────────────────────────────────────────────
print("\nTEST 1 — Sample separation")
x0 = torch.randn(1000, 1)
xt = x0.clone()
with torch.no_grad():
    for i in range(100):
        t_val = i / 100
        t = torch.full((1000, 1), t_val)
        q = backbone(xt, t)
        v, scores, _, _ = node(q)
        xt = xt + v / 100
samples = xt.numpy().flatten()
left  = (samples < -2).sum()
right = (samples > 2).sum()
mid   = ((samples > -2) & (samples < 2)).sum()
print(f"  Below -2: {left}  Above +2: {right}  Middle: {mid}")
if left > 400 and right > 400 and mid < 50:
    print("  PASS")
else:
    print("  FAIL — separation insufficient")

# ── TEST 2: Score sharpening over t ───────────────────────────────────────
print("\nTEST 2 — Score sharpening over t")
print("  Expected: scores uncertain early, confident late")
x0 = torch.randn(500, 1)
xt = x0.clone()
score_log = []
with torch.no_grad():
    for i in range(100):
        t_val = i / 100
        t = torch.full((500, 1), t_val)
        q = backbone(xt, t)
        v, scores, _, _ = node(q)
        confidence = (scores.max(dim=1).values).mean().item()
        score_log.append(confidence)
        xt = xt + v / 100
early_conf  = sum(score_log[:10])  / 10
mid_conf    = sum(score_log[45:55]) / 10
late_conf   = sum(score_log[90:])  / 10
print(f"  Confidence at t=0.0-0.1: {early_conf:.4f}")
print(f"  Confidence at t=0.4-0.5: {mid_conf:.4f}")
print(f"  Confidence at t=0.9-1.0: {late_conf:.4f}")
if late_conf > mid_conf > early_conf:
    print("  PASS — scores sharpen correctly over t")
elif late_conf > early_conf:
    print("  PARTIAL — some sharpening but not monotonic")
else:
    print("  FAIL — scores not sharpening over t")

# ── TEST 3: Random key initialization ─────────────────────────────────────
print("\nTEST 3 — Does separation survive random key init")
print("  Testing 5 random initializations...")
passes = 0
for trial in range(5):
    node_rand = RoutingNode(query_dim=32, v_dim=1)
    node_rand.load_state_dict(torch.load("node.pt", weights_only=True))
    with torch.no_grad():
        node_rand.k_left.normal_(0, 1)
        node_rand.k_right.normal_(0, 1)
        node_rand.k_right.data = node_rand.k_right - (
            node_rand.k_right @ node_rand.k_left /
            (node_rand.k_left @ node_rand.k_left)
        ) * node_rand.k_left
        node_rand.k_left.data  = node_rand.k_left  / node_rand.k_left.norm()
        node_rand.k_right.data = node_rand.k_right / node_rand.k_right.norm()
    node_rand.eval()
    x0 = torch.randn(500, 1)
    xt = x0.clone()
    with torch.no_grad():
        for i in range(100):
            t_val = i / 100
            t = torch.full((500, 1), t_val)
            q = backbone(xt, t)
            v, scores, _, _ = node_rand(q)
            xt = xt + v / 100
    s = xt.numpy().flatten()
    l = (s < -2).sum()
    r = (s > 2).sum()
    m = ((s > -2) & (s < 2)).sum()
    result = "PASS" if l > 200 and r > 200 and m < 100 else "FAIL"
    if result == "PASS":
        passes += 1
    print(f"  Trial {trial+1}: below -2: {l}  above +2: {r}  middle: {m}  — {result}")
if passes >= 4:
    print("  OVERALL PASS — separation robust to key initialization")
elif passes >= 2:
    print("  PARTIAL — separation partially robust")
else:
    print("  FAIL — separation depends on key initialization")

# ── TEST 4: Key initialization dependency ─────────────────────────────────
print("\nTEST 4 — Key similarity check")
sim = F.cosine_similarity(
    node.k_left.unsqueeze(0),
    node.k_right.unsqueeze(0)
).item()
print(f"  Current key similarity: {sim:.4f}")
if sim < -0.7:
    print("  PASS — keys well separated")
elif sim < 0:
    print("  PARTIAL — keys separated but weakly")
else:
    print("  FAIL — keys not separated")

# ── FINAL VERDICT ──────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("FINAL VERDICT")
print("If Test 1 PASS and Test 3 PASS — Gate 1 fully done.")
print("If Test 2 FAIL — score sharpening needs fixing before Gate 2.")
print("If Test 3 FAIL — key initialization is doing real work. Must fix.")
print("=" * 50)