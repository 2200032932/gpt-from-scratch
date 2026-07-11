import torch
import torch.nn as nn
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

block_size = 8
batch_size = 4
embedding_dim = 32

x = data[:block_size]

print(x)

embedding = nn.Embedding(
    num_embeddings=len(chars),
    embedding_dim=embedding_dim
)

embedded = embedding(x)

print("=" * 40)
print("Embedding Layer")
print("=" * 40)

print("Input Shape:")
print(x.shape)

print("\nEmbedding Shape:")
print(embedded.shape)


print("\nEmbedding for first token:\n")
print(embedded[0])

print(embedding.weight.shape)