#!/usr/bin/env python3
"""
Groove MIDI Dataset Reference Preparer - Robust Version
- Fixes FluidSynth argument order.
- Disables ALSA audio drivers to prevent crashes.
- Ensures strict cleanup of failed tracks.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
from datasets import load_dataset
import tqdm

def check_dependencies():
    """Check for system tools."""
    for tool in ["fluidsynth", "ffmpeg"]:
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} is not installed. Run: sudo apt install {tool}")
            sys.exit(1)

def main():
    # --- CONFIGURATION ---
    NUM_SAMPLES = 500 
    # Use absolute path for workspace to avoid 'File not found' errors
    BASE_DIR = Path("/home/emanuelebattistoni/Documents/workspace/MIDI-LLM")
    OUTPUT_DIR = BASE_DIR / "groove_reference_dataset"
    SOUNDFONT_PATH = BASE_DIR / "soundfonts/FluidR3_GM/FluidR3_GM.sf2"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ----------------------

    check_dependencies()

    if not SOUNDFONT_PATH.exists():
        print(f"ERROR: SoundFont not found at: {SOUNDFONT_PATH}")
        print("Please verify the file exists at that exact path.")
        return

    print(f"Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset("schismaudio/groove-midi-dataset", split="train")
        dataset = dataset.select_columns(["midi"])
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    subset = dataset.shuffle(seed=42).select(range(min(NUM_SAMPLES, len(dataset))))
    print(f"Processing {len(subset)} tracks...")

    success_count = 0

    for idx, item in enumerate(tqdm.tqdm(subset, desc="Synthesizing")):
        temp_midi = OUTPUT_DIR / f"temp_{idx}.mid"
        temp_wav = OUTPUT_DIR / f"temp_{idx}.wav"
        final_mp3 = OUTPUT_DIR / f"groove_ref_{idx}.mp3"

        try:
            # 1. Save MIDI bytes
            midi_data = item['midi']
            midi_bytes = midi_data['bytes'] if isinstance(midi_data, dict) else midi_data
            
            with open(temp_midi, "wb") as f:
                f.write(midi_bytes)

            # 2. Synthesize MIDI to WAV
            # Correct order: [flags] -> [output] -> [sample rate] -> [soundfont] -> [input]
            synth_cmd = [
                "fluidsynth", 
                "-ni",                # No interactive mode
                "-a", "null",         # Use NULL audio driver (prevents ALSA errors)
                "-F", str(temp_wav),  # Output file
                "-r", "44100",        # Sample rate
                str(SOUNDFONT_PATH),  # SoundFont
                str(temp_midi)        # Input MIDI
            ]
            
            synth_result = subprocess.run(synth_cmd, capture_output=True, text=True)

            if synth_result.returncode != 0:
                # Print error but do not save anything
                print(f"\n[SKIP] Track {idx} FluidSynth Error: {synth_result.stderr.strip()}")
                continue 

            # 3. Convert WAV to MP3
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temp_wav), 
                "-q:a", "0", "-map", "a", str(final_mp3)
            ]
            
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if ffmpeg_result.returncode != 0:
                print(f"\n[SKIP] Track {idx} FFmpeg Error: {ffmpeg_result.stderr.strip()}")
                continue

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Unexpected error on Track {idx}: {e}")
            if final_mp3.exists():
                final_mp3.unlink()
        
        finally:
            # Strict Cleanup: Always delete temp files
            if temp_midi.exists():
                temp_midi.unlink()
            if temp_wav.exists():
                temp_wav.unlink()

    print(f"\nFinished. Successfully created {success_count} MP3 files.")
    print(f"Reference folder: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()