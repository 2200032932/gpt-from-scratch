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
num_heads = 4
head_size = embedding_dim // num_heads
class FeedForward(nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            nn.ReLU(),

            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )

        )

    def forward(self, x):
        return self.net(x)
    

class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.mha = MultiHeadAttention(embedding_dim, num_heads)

        self.ffn = FeedForward(embedding_dim)

        self.ln1 = nn.LayerNorm(embedding_dim)

        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        x = x + self.mha(self.ln1(x))

        x = x + self.ffn(self.ln2(x))

        return x





class Head(nn.Module):

    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.key = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            head_size,
            bias=False
        )

    def forward(self, x):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)

        
        scores = scores / (self.head_size ** 0.5)
        
        if x.dim() == 2:
            T = x.size(0)
        else:
            T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        weights = F.softmax(
            scores,
            dim=-1
        )

        out = weights @ V

        return out



projection = nn.Linear(
    embedding_dim,
    embedding_dim
)

class MultiHeadAttention(nn.Module):

    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList([
            Head(embedding_dim, self.head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        out = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        out = self.projection(out)

        return out



if __name__ == "__main__":
    block = TransformerBlock(embedding_dim)
    result = block(final_embeddings)
    print(result.shape)