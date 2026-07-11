from pathlib import Path

print("=" * 40)
print("📖 Reading Dataset...\n")

# Project root
project_root = Path(__file__).resolve().parent.parent

# Dataset path
file_path = project_root / "data" / "input.txt"

# Read file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Dataset loaded successfully!\n")

print(f"Characters: {len(text)}\n")

print("First 300 characters:\n")

print(text[:300])

print("\n" + "=" * 40)

print("\nExtra Information")
print("-" * 30)
print(f"Type of text: {type(text)}")
print(f"Total Lines: {len(text.splitlines())}")
print(f"Total Words: {len(text.split())}")