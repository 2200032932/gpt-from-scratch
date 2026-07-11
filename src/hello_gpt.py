import os
import sys


def greet():
    name = "Rishi"

    print("=" * 40)
    print("🚀 Welcome to GPT From Scratch!")
    print(f"Hello, {name}!")
    print(f"Python Version: {sys.version.split()[0]}")
    print("Current Working Directory:")
    print(os.getcwd())
    print()
    print("Everything is working correctly!")
    print("=" * 40)


greet()