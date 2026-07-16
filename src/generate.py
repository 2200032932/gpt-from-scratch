import torch

from gpt_model import GPT, itos


def decode(tokens):
    return "".join(itos[i] for i in tokens)

model = GPT()

model.load_state_dict(
    torch.load("gpt_model.pth")
)

model.eval()

context = torch.zeros(
    (1, 1),
    dtype=torch.long
)

generated = model.generate(
    context,
    max_new_tokens=300
)

print(
    decode(generated[0].tolist())
)
