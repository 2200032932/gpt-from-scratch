import torch
import torch.nn as nn
from pathlib import Path
from transformer_block import TransformerBlock
import torch.nn.functional as F


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

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)

        return idx


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding(idx)

        positions = torch.arange(T)

        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
    

            loss = F.cross_entropy(logits, targets)

        return logits, loss   

    def decode(tokens):
        return "".join([itos[i] for i in tokens])

if __name__ == "__main__":

    model = GPT()

    idx = data[:block_size].unsqueeze(0)
    targets = data[1:block_size + 1].unsqueeze(0)
    logits, loss = model(idx, targets)

    print("Logits Shape:", logits.shape)

    print("Loss:", loss.item())