import torch

x = torch.tensor([[]])

y = torch.tensor([[]])

cos_layer = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
print(cos_layer(x, y))