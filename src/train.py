import torch
import torch.optim as optim

from gpt_model import GPT, data, block_size, itos

def decode(tokens):
    return "".join([itos[i] for i in tokens])

batch_size = 32
max_iters = 5000
learning_rate = 3e-4
eval_interval = 500

model = GPT()

optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

def get_batch():

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


for iter in range(max_iters):

    xb, yb = get_batch()

    logits, loss = model(xb, yb)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if iter % eval_interval == 0:
        print(f"Iteration {iter} | Loss: {loss.item():.4f}")

        context = torch.zeros((1, 1), dtype=torch.long)

generated = model.generate(
    context,
    max_new_tokens=300
)

print()
print(decode(generated[0].tolist()))

torch.save(
    model.state_dict(),
    "gpt_model.pth"
)

print("Model saved successfully!")