import torch
from tripletface.core.model import Encoder

model = Encoder(64)
weight = torch.load("model.pt")['model']
model.load_state_dict(weight)

input = torch.randn((1, 3, 224, 224)).float()
module = torch.jit.trace(model, input, check_trace=False)
torch.jit.save(module, "JIT_model.pt")