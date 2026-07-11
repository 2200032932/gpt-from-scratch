import torch
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "input.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

encoded_text = encode(text)

data = torch.tensor(encoded_text, dtype=torch.long)


block_size = 8

x = data[:block_size]
y = data[1:block_size + 1]

print("=" * 40)
print("Creating Training Samples\n")

print(f"Context Length (Block Size): {block_size}\n")

print("Input:")
print(x)

print("\nDecoded:")
print(''.join([itos[i.item()] for i in x]))

print("\nTarget:")
print(y)

print("\nDecoded:")
print(''.join([itos[i.item()] for i in y]))

print("\n" + "=" * 40)

print("\nNext Token Prediction:\n")

for t in range(block_size):
    context = x[:t + 1]
    target = y[t]

    print(f"Input : {context.tolist()}")
    print(f"Target: {target.item()} ('{itos[target.item()]}')")