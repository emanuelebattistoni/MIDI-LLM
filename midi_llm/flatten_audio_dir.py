"""
Flatten Audio Directory Script
Extracts all audio files from subdirectories and copies them into a single flat folder.
This is required for FAD evaluation when files are nested in timestamped folders.
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def main():
    # 1. Command Line Interface configuration
    parser = argparse.ArgumentParser(
        description="Flatten a nested directory by extracting all audio files into a single folder."
    )
    parser.add_argument(
        "--src", 
        type=str, 
        required=True, 
        help="Source directory containing nested subfolders (e.g., eval_results)"
    )
    parser.add_argument(
        "--dest", 
        type=str, 
        default="./data/lora2_eval_set2", 
        help="Target directory for flat audio storage (default: fad_eval_set)"
    )
    parser.add_argument(
        "--move", 
        action="store_true", 
        help="If set, move files instead of copying them (saves disk space)"
    )

    args = parser.parse_args()

    # 2. Path setup
    src_root = Path(args.src).resolve()
    dest_root = Path(args.dest).resolve()

    if not src_root.exists():
        print(f"Error: Source directory '{src_root}' not found.")
        return

    # Create destination folder if it doesn't exist
    dest_root.mkdir(parents=True, exist_ok=True)

    # 3. Finding audio files
    # We look for .mp3 and .wav files in all subdirectories
    audio_extensions = [".mp3", ".wav"]
    all_files = []
    for ext in audio_extensions:
        all_files.extend(list(src_root.rglob(f"*{ext}")))

    if not all_files:
        print(f"No audio files found in '{src_root}'.")
        return

    print(f"Found {len(all_files)} audio files. Starting process...")

    # 4. Processing files
    success_count = 0
    # Use tqdm for progress feedback
    for file_path in tqdm(all_files, desc="Processing"):
        # Create a unique filename if duplicates exist across different subfolders
        # We use the parent folder name as a prefix for safety
        prefix = file_path.parent.name
        new_filename = f"{prefix}_{file_path.name}"
        target_path = dest_root / new_filename

        try:
            if args.move:
                shutil.move(str(file_path), str(target_path))
            else:
                shutil.copy2(str(file_path), str(target_path))
            success_count += 1
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")

    print(f"\nOperation completed!")
    print(f"Successfully processed {success_count} files into: {dest_root}")

if __name__ == "__main__":
    main()