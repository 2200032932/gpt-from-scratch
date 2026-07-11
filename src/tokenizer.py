from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
file_path = project_root / "data" / "input.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(list(set(text)))    
vocab_size = len(chars)    

print(f"Vocabulary Size: {vocab_size}")
print(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print("\nCharacter to Integer:")
print(stoi['A'])

print("\nInteger to Character:")
print(itos[stoi['A']])

def encode(s):
    return [stoi[c] for c in s]
print(encode("Hello"))
def decode(tokens):
    return ''.join([itos[i] for i in tokens])

encoded = encode("Hello GPT")

print(encoded)

decoded = decode(encoded)

print(decoded)