import torch
import torch.nn as nn
from pathlib import Path
from transformer_block import TransformerBlock


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
chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(stoi)


class GPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = nn.Embedding(
            block_size,
            embedding_dim
        )

        self.blocks = nn.Sequential(
            TransformerBlock(embedding_dim),
            TransformerBlock(embedding_dim),
            TransformerBlock(embedding_dim)
        )

        self.ln = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, idx):
        B, T = idx.shape

        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(T)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.lm_head(x)

        return logits     


if __name__ == "__main__":

    model = GPT()

    idx = data[:block_size].unsqueeze(0)

    logits = model(idx)

    print(logits.shape)    