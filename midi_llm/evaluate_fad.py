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
from pathlib import Path

#START PATCH TO FIX PYTORCH HUB BUG (VGGish)
import torch
_original_hub_load = torch.hub.load

def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
    # Force PyTorch Hub to use the 'master' branch for torchvggish
    # to avoid the "Found default branches ['master', 'main']" error
    if repo_or_dir == 'harritaylor/torchvggish':
        repo_or_dir = 'harritaylor/torchvggish:master' 
    return _original_hub_load(repo_or_dir, model, *args, **kwargs)

torch.hub.load = _patched_hub_load
#END PATCH

try:
    from frechet_audio_distance import FrechetAudioDistance
except ImportError:
    print("Error: frechet_audio_distance library not found.")
    print("Install it using: pip install frechet_audio_distance")
    sys.exit(1)

#LOADING ANIMATION CLASS
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


def main():
    # 1. Command Line Interface configuration
    parser = argparse.ArgumentParser(
        description="Calculate FAD between two directories of audio files (e.g., MP3/WAV).",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Usage example:
  python evaluate_fad.py --ref_dir ./real_music --eval_dir ./generated_music
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

    # 4. Frechet Audio Distance Calculation
    print(f"\n Reference: {ref_path.absolute()}")
    print(f" Evaluation: {eval_path.absolute()}")
    print("(This operation may take a few minutes depending on the number of files and your GPU...)\n")

    try:
        # Use the spinner context manager to provide visual feedback during the blocking FAD computation
        with ProgressSpinner():
            fad_score = frechet.score(str(ref_path), str(eval_path))
            
        print(f"\n{'='*50}")
        print(f" FAD SCORE RESULT: {fad_score:.4f}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\nUnexpected error during FAD calculation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()