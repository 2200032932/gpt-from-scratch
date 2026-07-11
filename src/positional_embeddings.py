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
embedding_dim = 32

x = data[:block_size]

print("Token IDs:")
print(x)

token_embedding = nn.Embedding(
    len(stoi),
    embedding_dim
)

token_vectors = token_embedding(x)

positions = torch.arange(block_size)

print(positions)


position_embedding = nn.Embedding(
    block_size,
    embedding_dim
)

position_vectors = position_embedding(positions)

print(position_vectors.shape)

final_embeddings = token_vectors + position_vectors

print("=" * 40)
print("Positional Embeddings")
print("=" * 40)

print("Token Embedding Shape:")
print(token_vectors.shape)

print()

print("Position Embedding Shape:")
print(position_vectors.shape)

print()

print("Final Embedding Shape:")
print(final_embeddings.shape)

print("=" * 40)

print("\nToken Vector:\n")
print(token_vectors[0])

print("\nPosition Vector:\n")
print(position_vectors[0])

print("\nFinal Vector:\n")
print(final_embeddings[0])
