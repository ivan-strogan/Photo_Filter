#!/usr/bin/env python3
"""Debug file detection."""

import re
from pathlib import Path

# Test the regex pattern
FILENAME_PATTERN = r"IMG_(\d{8})_(\d{6})(?:_\d+)?\.(JPG|MOV|PNG|jpg|mov|png)$"
pattern = re.compile(FILENAME_PATTERN, re.IGNORECASE)

iphone_dir = Path("Sample_Photos/iPhone Automatic")

print("=== Debugging File Detection ===")
print(f"Looking in: {iphone_dir}")
print(f"Directory exists: {iphone_dir.exists()}")

if iphone_dir.exists():
    files = list(iphone_dir.iterdir())
    print(f"Total files in directory: {len(files)}")

    print("\nFirst 10 files:")
    for i, file_path in enumerate(files[:10]):
        if file_path.is_file():
            filename = file_path.name
            match = pattern.match(filename)
            print(f"  {filename} -> Match: {bool(match)}")
            if match:
                print(f"    Groups: {match.groups()}")

    # Test different patterns
    print("\nTesting different patterns:")
    test_patterns = [
        r"IMG_(\d{8})_(\d{6})\.(JPG|MOV|PNG)$",
        r"IMG_(\d{8})_(\d{6})(?:_\d+)?\.(JPG|MOV|PNG)$",
        r"IMG_(\d{8})_(\d{6}).*\.(JPG|MOV|PNG)$",
    ]

    for pattern_str in test_patterns:
        test_pattern = re.compile(pattern_str, re.IGNORECASE)
        matches = 0
        for file_path in files[:20]:  # Test first 20 files
            if file_path.is_file() and test_pattern.match(file_path.name):
                matches += 1
        print(f"  Pattern '{pattern_str}': {matches} matches")