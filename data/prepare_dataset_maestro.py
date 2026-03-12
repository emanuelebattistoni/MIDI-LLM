import os
import subprocess
import sys
import shutil
import argparse
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

def check_dependencies():
    """Verify that required external CLI tools are installed."""
    for tool in ["fluidsynth", "ffmpeg"]:
        if shutil.which(tool) is None:
            print(f"ERROR: '{tool}' not found. Please install it (e.g., sudo apt install {tool})")
            sys.exit(1)

def download_file(url, filepath):
    """Download a file with a visible progress bar."""
    if filepath.exists():
        print(f"File {filepath.name} already exists locally.")
        return
    
    print(f"Downloading dataset archive...")
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading") as t:
        def reporthook(blocknum, blocksize, totalsize):
            t.total = totalsize
            t.update(blocknum * blocksize - t.n)
        urllib.request.urlretrieve(url, filepath, reporthook)

def main():
    # 1. Setup Command Line Arguments for maximum flexibility
    parser = argparse.ArgumentParser(description="MIDI Dataset Extractor and MP3 Synthesizer")
    
    parser.add_argument("--limit", type=int, default=500, help="Exact number of MP3 files to generate")
    parser.add_argument("--output_dir", type=str, default="./data/maestro_reference_dataset", help="Path to the output directory")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the downloaded dataset archive")
    parser.add_argument("--dataset_url", type=str, default="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip", help="URL of the dataset ZIP file")
    parser.add_argument("--soundfont", type=str, default="./soundfonts/FluidR3_GM/FluidR3_GM.sf2", help="Path to the SoundFont (.sf2) file")
    parser.add_argument("--sample_rate", type=str, default="44100", help="Sample rate for audio synthesis (e.g., 44100, 48000)")
    parser.add_argument("--audio_quality", type=str, default="2", help="FFmpeg audio quality scale (q:a)")

    args = parser.parse_args()

    # 2. Resolve paths absolutely
    output_dir = Path(args.output_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    soundfont_path = Path(args.soundfont).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    check_dependencies()

    if not soundfont_path.exists():
        print(f"\nCRITICAL ERROR: SoundFont file not found at:\n{soundfont_path}")
        print("Please provide a valid path using the --soundfont argument.")
        sys.exit(1)

    print(f"--- MIDI DATASET EXTRACTOR ---")
    print(f"Goal: Generate up to {args.limit} MP3 tracks from native MIDI files")

    # Extract filename from URL dynamically
    zip_filename = Path(args.dataset_url).name
    zip_path = data_dir / zip_filename
    
    try:
        download_file(args.dataset_url, zip_path)
    except Exception as e:
        print(f"Error during download: {e}")
        return

    success_count = 0
    pbar = tqdm(total=args.limit, desc="Overall Progress", unit="track")

    try:
        # Open the zip without extracting everything to disk to save space
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Find all .midi or .mid files in the archive
            midi_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.mid', '.midi'))]
            
            for file_info in midi_files:
                if success_count >= args.limit:
                    break

                # Create a clean, truncated name
                original_name = Path(file_info).stem.replace(" ", "_")
                current_id = f"track_{success_count + 1:04d}_{original_name[:30]}"
                
                pbar.set_postfix_str(f"Processing: {current_id}")

                temp_midi = output_dir / f"{current_id}.mid"
                temp_wav = output_dir / f"{current_id}.wav"
                final_mp3 = output_dir / f"{current_id}.mp3"

                try:
                    # Extract the raw MIDI file from the archive
                    with zip_ref.open(file_info) as source, open(temp_midi, "wb") as target:
                        shutil.copyfileobj(source, target)

                    # Audio synthesis using variables instead of hard-coded values
                    subprocess.run([
                        "fluidsynth", "-ni", "-a", "null", "-F", str(temp_wav), 
                        "-r", args.sample_rate, str(soundfont_path), str(temp_midi)
                    ], capture_output=True, check=True)
                    
                    subprocess.run([
                        "ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_wav), 
                        "-q:a", args.audio_quality, "-map", "a", str(final_mp3)
                    ], capture_output=True, check=True)
                    
                    success_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    pbar.write(f"\n Error on {current_id}: {e}")
                    continue
                finally:
                    # Ensure temporary files are cleaned up to prevent disk clutter
                    if temp_midi.exists(): temp_midi.unlink()
                    if temp_wav.exists(): temp_wav.unlink()

    except Exception as e:
        print(f"\n Error opening ZIP file: {e}")
    finally:
        pbar.close()

    print(f"\nOperation completed successfully! {success_count} MP3 files saved to {output_dir}")

if __name__ == "__main__":
    main()