import torch

scripted = torch.jit.load("mobilefacenet_scripted.pt", map_location="cpu")
torch.save(scripted.state_dict(), "mobilefacenet_weights.pth")
