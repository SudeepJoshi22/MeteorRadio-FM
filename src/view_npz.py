# Save this as view_npz.py (replace the old one)

import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python3 view_npz.py <filename.npz>")
    sys.exit(1)

filename = sys.argv[1]

try:
    with np.load(filename) as data:
        print(f"File: {filename}")
        print("Arrays inside:", data.files)
        print()

        for key in data.files:
            arr = data[key]
            print(f"{key}:")
            print(f" size: {arr.shape}")

            # If it's a scalar (shape () )
            if arr.shape == ():
                print(arr.item())   # prints the single value cleanly
            else:
                print(arr)          # prints the whole array simply

            print()  # blank line between arrays

except Exception as e:
    print(f"Error: {e}")
