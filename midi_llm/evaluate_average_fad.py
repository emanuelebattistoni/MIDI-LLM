"""
Calculate Frechet Audio Distance (FAD)
Compares two directories of audio files (e.g., MP3/WAV) to evaluate generation quality.
"""

import os
import sys
import argparse
import threading
import itertools
import time
import tempfile
import shutil
from pathlib import Path

# --- START PATCH TO FIX PYTORCH HUB BUG (VGGish) ---
import torch
_original_hub_load = torch.hub.load

def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
    # Force PyTorch Hub to use the 'master' branch for torchvggish
    # to avoid the "Found default branches ['master', 'main']" error
    if repo_or_dir == 'harritaylor/torchvggish':
        repo_or_dir = 'harritaylor/torchvggish:master' 
    return _original_hub_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = _patched_hub_load
# --- END PATCH ---

try:
    from frechet_audio_distance import FrechetAudioDistance
except ImportError:
    print("Error: frechet_audio_distance library not found.")
    print("Install it using: pip install frechet_audio_distance")
    sys.exit(1)

# --- LOADING ANIMATION CLASS ---
class ProgressSpinner:
    """A context manager class that provides a background loading animation."""
    def __init__(self, message="Calculating Frechet Audio Distance..."):
        # Use Braille characters to create a modern, smooth spinner effect
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.delay = 0.1
        self.busy = False
        self.message = message
        self.thread = None

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(f"\r{self.message} {next(self.spinner)}")
            sys.stdout.flush()
            time.sleep(self.delay)

    def __enter__(self):
        self.busy = True
        self.thread = threading.Thread(target=self.spinner_task)
        self.thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.busy = False
        if self.thread:
            self.thread.join()
        # Clear the terminal line and print the completion status
        sys.stdout.write(f"\r{self.message} Completed!       \n")
        sys.stdout.flush()

# --- HELPER: CREATE TEMPORARY DIRECTORY WITH FILE LIMIT ---
def create_limited_dir(source_dir: Path, limit: int) -> str:
    """
    Creates a temporary directory with symlinks to a limited number of audio files.
    This prevents memory/disk duplication while allowing the FAD library to process a subset.
    """
    temp_dir = tempfile.mkdtemp(prefix="fad_eval_")
    
    # Collect valid audio files and sort them to ensure deterministic behavior
    valid_extensions = ['.mp3', '.wav', '.flac']
    files = sorted([f for f in source_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions])
    
    limited_files = files[:limit]
    
    for f in limited_files:
        dest_path = Path(temp_dir) / f.name
        try:
            # Create a symbolic link using the ABSOLUTE path to prevent broken links
            os.symlink(f.absolute(), dest_path)
        except OSError:
            # Fallback to physical copy if the OS (e.g., Windows) blocks symlinks
            shutil.copy2(f.absolute(), dest_path)
            
    return temp_dir


def main():
    # 1. Command Line Interface configuration
    parser = argparse.ArgumentParser(
        description="Calculate FAD between two directories of audio files (e.g., MP3/WAV).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage examples:
  Evaluate all files:
    python evaluate_fad.py --ref_dir ./real_music --eval_dir ./generated_music
    
  Evaluate only the first 50 files:
    python evaluate_fad.py --ref_dir ./real_music --eval_dir ./generated_music --limit 50
        """
    )
    
    parser.add_argument(
        "--ref_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing reference audio files (real background)."
    )
    
    parser.add_argument(
        "--eval_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing audio files to evaluate (AI generated)."
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Maximum number of audio files to evaluate from each directory (optional)."
    )

    args = parser.parse_args()

    # 2. Path validation
    ref_path = Path(args.ref_dir)
    eval_path = Path(args.eval_dir)

    if not ref_path.exists() or not ref_path.is_dir():
        print(f"Error: The reference directory '{ref_path}' does not exist or is not a valid directory.")
        sys.exit(1)

    if not eval_path.exists() or not eval_path.is_dir():
        print(f"Error: The evaluation directory '{eval_path}' does not exist or is not a valid directory.")
        sys.exit(1)

    # 3. FAD Model Initialization
    print("\nInitializing VGGish model for FAD calculation...")
    try:
        # Note: using standard frechet_audio_distance parameters for VGGish
        frechet = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
    except Exception as e:
        print(f"Error during FAD model initialization: {e}")
        sys.exit(1)

    # 4. Limit Handling & Setup Paths
    active_ref_path = str(ref_path)
    active_eval_path = str(eval_path)
    temp_dirs_to_clean = []

    if args.limit is not None and args.limit > 0:
        print(f"Applying file limit: Evaluating only the first {args.limit} files per directory.")
        
        active_ref_path = create_limited_dir(ref_path, args.limit)
        active_eval_path = create_limited_dir(eval_path, args.limit)
        
        temp_dirs_to_clean.extend([active_ref_path, active_eval_path])

    print(f"\n Reference: {active_ref_path}")
    print(f" Evaluation: {active_eval_path}")
    print("(This operation may take a few minutes depending on the number of files and your GPU...)\n")

    # 5. Frechet Audio Distance Calculation
    try:
        # Use the spinner context manager to provide visual feedback during the blocking FAD computation
        with ProgressSpinner():
            fad_score = frechet.score(active_ref_path, active_eval_path)
            
        print(f"\n{'='*50}")
        print(f" FAD SCORE RESULT: {fad_score:.4f}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\nUnexpected error during FAD calculation: {e}")
        sys.exit(1)
        
    finally:
        # 6. Cleanup Temporary Directories
        for t_dir in temp_dirs_to_clean:
            if os.path.exists(t_dir):
                shutil.rmtree(t_dir, ignore_errors=True)

if __name__ == "__main__":
    main()