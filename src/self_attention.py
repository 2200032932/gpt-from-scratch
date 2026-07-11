import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "input.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))


stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

encoded_text = encode(text)

data = torch.tensor(encoded_text, dtype=torch.long)

block_size = 8
embedding_dim = 32

x = data[:block_size]



token_embedding = nn.Embedding(
    len(stoi),
    embedding_dim
)

token_vectors = token_embedding(x)

positions = torch.arange(block_size)


position_embedding = nn.Embedding(
    block_size,
    embedding_dim
)

position_vectors = position_embedding(positions)


final_embeddings = token_vectors + position_vectors

head_size = 16

query = nn.Linear(
    embedding_dim,
    head_size,
    bias=False
)

key = nn.Linear(
    embedding_dim,
    head_size,
    bias=False
)

value = nn.Linear(
    embedding_dim,
    head_size,
    bias=False
)

Q = query(final_embeddings)
K = key(final_embeddings)
V = value(final_embeddings)

print("=" * 40)
print("Query, Key, Value")
print("=" * 40)

print("Q Shape:", Q.shape)
print("K Shape:", K.shape)
print("V Shape:", V.shape)

scores = Q @ K.transpose(-2, -1)
print("Scores Shape:", scores.shape)

scores = scores / (head_size ** 0.5)
mask = torch.tril(torch.ones(block_size, block_size))

print(mask)

scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)

print(weights.shape)
print(weights.sum(dim=-1))


out = weights @ V

print(out.shape)

