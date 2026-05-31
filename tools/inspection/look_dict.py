import torch

checkpoint = torch.load("bc_distill_best.pt")

# 打印所有的 Key，看看层名对不对
print("模型层名列表:", checkpoint.keys())

# 查看某一层的形状（比如第一层的权重形状）
print("第一层权重形状:", checkpoint["net.0.weight"].shape)
# 应该是 [256, obs_dim]

