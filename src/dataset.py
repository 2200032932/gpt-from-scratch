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

type(encoded_text)
data = torch.tensor(encoded_text, dtype=torch.long)


print("=" * 40)
print("📚 GPT Dataset Preparation\n")

print(f"Vocabulary Size: {vocab_size}\n")

print(f"Dataset Shape: {data.shape}\n")

print("First 20 Token IDs:")
print(data[:20])

print("\nFirst 20 Decoded Characters:")

decoded = ''.join([itos[i.item()] for i in data[:20]])
print(decoded)

print("\nData Type:")
print(data.dtype)

print("\nDevice:")
print(data.device)

print("=" * 40)