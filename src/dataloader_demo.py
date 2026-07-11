import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "input.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 8
batch_size = 4

class GPTDataset(Dataset):

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y
    
dataset = GPTDataset(data, block_size)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

print("=" * 40)
print("Mini Batch Example")
print("=" * 40)

for x, y in loader:
    print("Input Shape:", x.shape)
    print("Target Shape:", y.shape)

    print("\nInput Batch:")
    print(x)

    print("\nTarget Batch:")
    print(y)

    break

sample = x[0]

decoded = ''.join([itos[i.item()] for i in sample])

print("\nFirst Sample Decoded:")
print(decoded)

