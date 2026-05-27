# dotcheck.py
import torch
from backbone import Backbone
from node import RoutingNode

backbone = Backbone(input_dim=2, hidden_dim=64, output_dim=32)
backbone.load_state_dict(torch.load("backbone.pt", weights_only=True))
backbone.eval()

node = RoutingNode(query_dim=32, v_dim=1)
node.load_state_dict(torch.load("node.pt", weights_only=True))
node.eval()

x_left  = torch.full((10, 1), -3.5)
x_right = torch.full((10, 1),  3.5)
t_late  = torch.full((10, 1),  0.9)

with torch.no_grad():
    q_left  = backbone(x_left,  t_late)
    q_right = backbone(x_right, t_late)
    
    sl_left  = (q_left  * node.k_left).sum(-1)
    sr_left  = (q_left  * node.k_right).sum(-1)
    sl_right = (q_right * node.k_left).sum(-1)
    sr_right = (q_right * node.k_right).sum(-1)
    
    print(f"Left  sample: score_left={sl_left.mean():.4f}  score_right={sr_left.mean():.4f}")
    print(f"Right sample: score_left={sl_right.mean():.4f}  score_right={sr_right.mean():.4f}")

print(f"q_left  norm: {q_left.norm(dim=-1).mean():.4f}")
print(f"q_right norm: {q_right.norm(dim=-1).mean():.4f}")
print(f"k_left  norm: {node.k_left.norm():.4f}")
print(f"k_right norm: {node.k_right.norm():.4f}")

print(f"k_left  direction sim to q_left:  {torch.nn.functional.cosine_similarity(node.k_left.unsqueeze(0), q_left.mean(0, keepdim=True)).item():.4f}")
print(f"k_right direction sim to q_right: {torch.nn.functional.cosine_similarity(node.k_right.unsqueeze(0), q_right.mean(0, keepdim=True)).item():.4f}")



# in debug.py
x_left  = torch.full((100, 1), -3.5)
x_right = torch.full((100, 1),  3.5)
t_late  = torch.full((100, 1),  0.9)

with torch.no_grad():
    q_left  = backbone(x_left,  t_late)
    q_right = backbone(x_right, t_late)
    _, _, v_left_out,  _ = node(q_left)
    _,  _, _, v_right_out = node(q_right)
    print(f"v_left  for left  sample: {v_left_out.mean():.4f}")
    print(f"v_right for right sample: {v_right_out.mean():.4f}")


from data import sample_trajectory
_, x1, _, _, target_v = sample_trajectory(1000)
left_mask = (x1 < 0).squeeze()
print(f"Mean target_v for left samples: {target_v[left_mask].mean():.4f}")
print(f"Mean target_v for right samples: {target_v[~left_mask].mean():.4f}")