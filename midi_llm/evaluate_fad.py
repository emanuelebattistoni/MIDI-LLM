#!/usr/bin/env python3
"""
External script to calculate the Frechet Audio Distance (FAD) 
between a reference dataset (Real Music) and a generated dataset (AI Music).
Requirements: pip install fadtk
"""

import argparse
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Calculate FAD score between real and generated audio.")
    parser.add_argument("--reference", type=str, required=True, help="Path to the directory with REAL reference audio")
    parser.add_argument("--generated", type=str, required=True, help="Path to the directory with GENERATED AI audio")
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="clap-2023", 
        help="Embedding model to use (default: clap-2023. Other valid options: vggish, encodec-emb, wavlm-base)"
    )
    
    args = parser.parse_args()

    ref_dir = Path(args.reference)
    gen_dir = Path(args.generated)

    # Basic directory validation
    if not ref_dir.exists() or not gen_dir.exists():
        print("ERROR: One or both directories do not exist. Please check your paths.")
        sys.exit(1)

    # --- AUTO-FILTERING MAGIC ---
    # Cerca tutti i file .mp3 ricorsivamente all'interno della cartella generata
    mp3_files = list(gen_dir.rglob("*.mp3"))
    
    if not mp3_files:
        print(f"ERROR: No .mp3 files found in {gen_dir} or its subdirectories.")
        sys.exit(1)

    print("-" * 60)
    print("STARTING FRECHET AUDIO DISTANCE (FAD) CALCULATION")
    print("-" * 60)
    print(f"Reference Folder : {ref_dir.absolute()}")
    print(f"Generated Folder : {gen_dir.absolute()} (Found {len(mp3_files)} audio files)")
    print(f"Embedding Model  : {args.model}")
    print("Preparing audio files... This may take a few minutes.")
    print("-" * 60)

    # Crea una cartella temporanea sicura che si autodistruggerà alla fine
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio_dir = Path(temp_dir)
        
        # Copia tutti gli MP3 nella cartella temporanea rinominandoli per evitare sovrascritture
        for f in mp3_files:
            # Crea un nome univoco unendo il nome della cartella padre e del file (es: prompt_1_gen_1.mp3)
            safe_filename = f"{f.parent.name}_{f.name}"
            shutil.copy2(f, temp_audio_dir / safe_filename)

        try:
            # Construct and execute the fadtk command puntando alla cartella temporanea!
            cmd = ["fadtk", args.model, str(ref_dir), str(temp_audio_dir)]
            
            # Run the subprocess and capture output
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Output the result to terminal
            print("\nCALCULATION COMPLETE")
            print("FAD Output:")
            print(result.stdout)
            
            # Save results to a report file inside the original generated folder
            report_path = gen_dir / "REPORT_FAD.txt"
            with open(report_path, "w") as f:
                f.write("Frechet Audio Distance (FAD) Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"Reference Dataset: {ref_dir.name}\n")
                f.write(f"Embedding Model: {args.model}\n")
                f.write(f"Number of AI tracks analyzed: {len(mp3_files)}\n")
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