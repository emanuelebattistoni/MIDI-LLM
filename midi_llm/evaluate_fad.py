#!/usr/bin/env python3
"""
External script to calculate the Frechet Audio Distance (FAD) 
between a reference dataset (Real Music) and a generated dataset (AI Music).
Requirements: pip install fadtk
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calculate FAD score between real and generated audio.")
    parser.add_argument("--reference", type=str, required=True, help="Path to the directory with REAL reference audio")
    parser.add_argument("--generated", type=str, required=True, help="Path to the directory with GENERATED AI audio")
    parser.add_argument("--model", type=str, default="enc-dec", help="Embedding model to use (default: enc-dec, options: clap, vggish)")
    
    args = parser.parse_args()

    ref_dir = Path(args.reference)
    gen_dir = Path(args.generated)

    # Basic directory validation
    if not ref_dir.exists() or not gen_dir.exists():
        print("ERROR: One or both directories do not exist. Please check your paths.")
        sys.exit(1)

    print("-" * 60)
    print("STARTING FRECHET AUDIO DISTANCE (FAD) CALCULATION")
    print("-" * 60)
    print(f"Reference Folder : {ref_dir.absolute()}")
    print(f"Generated Folder : {gen_dir.absolute()}")
    print(f"Embedding Model  : {args.model}")
    print("This process may take a few minutes depending on the number of files...")
    print("-" * 60)

    try:
        # Construct and execute the fadtk command
        # Syntax: fadtk <model> <baseline_dir> <eval_dir>
        cmd = ["fadtk", args.model, str(ref_dir), str(gen_dir)]
        
        # Run the subprocess and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Output the result to terminal
        print("\nCALCULATION COMPLETE")
        print("FAD Output:")
        print(result.stdout)
        
        # Save results to a report file inside the generated folder
        report_path = gen_dir / "REPORT_FAD.txt"
        with open(report_path, "w") as f:
            f.write("Frechet Audio Distance (FAD) Report\n")
            f.write("=" * 40 + "\n")
            f.write(f"Reference Dataset: {ref_dir.name}\n")
            f.write(f"Embedding Model: {args.model}\n")
            f.write("-" * 40 + "\n")
            f.write(result.stdout)
            
        print(f"\nReport successfully saved to: {report_path}")
        
    except subprocess.CalledProcessError as e:
        print("\nERROR: fadtk execution failed.")
        print(e.stderr)
    except FileNotFoundError:
        print("\nERROR: 'fadtk' command not found. Please ensure it is installed (pip install fadtk).")

if __name__ == "__main__":
    main()