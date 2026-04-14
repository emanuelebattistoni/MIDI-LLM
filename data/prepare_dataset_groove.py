#!/usr/bin/env python3
"""
Groove MIDI Local Preparer
- Legge i file MIDI dalla cartella locale groove-midi-results (incluse sottocartelle).
- Sintetizza in MP3 usando FluidSynth.
"""

import os
import subprocess
import sys
import shutil
import random
from pathlib import Path
import tqdm

def check_dependencies():
    """Verifica la presenza di fluidsynth e ffmpeg."""
    for tool in ["fluidsynth", "ffmpeg"]:
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} non è installato. Esegui: sudo apt install {tool}")
            sys.exit(1)

def main():
    NUM_SAMPLES = 1000 
    BASE_DIR = Path("/home/emanuelebattistoni/Documents/workspace/MIDI-LLM")
    
    SOURCE_DIR = BASE_DIR / "data/groove-midi-dataset" 
    
    OUTPUT_DIR = BASE_DIR / "groove_reference_dataset_augmented"
    SOUNDFONT_PATH = BASE_DIR / "soundfonts/FluidR3_GM/FluidR3_GM.sf2"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ----------------------

    check_dependencies()

    if not SOURCE_DIR.exists():
        print(f"ERROR: Cartella sorgente non trovata: {SOURCE_DIR}")
        return

    if not SOUNDFONT_PATH.exists():
        print(f"ERROR: SoundFont non trovato a: {SOUNDFONT_PATH}")
        return

    print(f"Ricerca file MIDI in {SOURCE_DIR}...")
    all_midi_files = list(SOURCE_DIR.rglob("*.mid"))
    
    if not all_midi_files:
        print("Nessun file .mid trovato nella cartella sorgente.")
        return

    print(f"Trovati {len(all_midi_files)} file MIDI.")

    random.seed(42) # Per riproducibilità
    if len(all_midi_files) > NUM_SAMPLES:
        selected_files = random.sample(all_midi_files, NUM_SAMPLES)
    else:
        selected_files = all_midi_files
    
    print(f"Inizio sintesi di {len(selected_files)} tracce...")

    success_count = 0

    for idx, midi_path in enumerate(tqdm.tqdm(selected_files, desc="Sintetizzando")):
        temp_wav = OUTPUT_DIR / f"temp_{idx}.wav"
        final_mp3 = OUTPUT_DIR / f"groove_ref_{idx}.mp3"

        try:
            synth_cmd = [
                "fluidsynth", 
                "-ni",                # No interactive mode
                "-a", "null",         # NULL audio driver (per ambienti senza scheda audio)
                "-F", str(temp_wav),  # File di output WAV
                "-r", "44100",        # Sample rate
                str(SOUNDFONT_PATH),  # SoundFont
                str(midi_path)        # File MIDI originale
            ]
            
            synth_result = subprocess.run(synth_cmd, capture_output=True, text=True)

            if synth_result.returncode != 0:
                print(f"\n[SKIP] Errore FluidSynth su {midi_path.name}: {synth_result.stderr.strip()}")
                continue 

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temp_wav), 
                "-q:a", "0", "-map", "a", str(final_mp3)
            ]
            
            ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

            if ffmpeg_result.returncode != 0:
                print(f"\n[SKIP] Errore FFmpeg su {midi_path.name}: {ffmpeg_result.stderr.strip()}")
                continue

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Errore inaspettato su {midi_path.name}: {e}")
        
        finally:
            if temp_wav.exists():
                temp_wav.unlink()

    print(f"\nOperazione completata!")
    print(f"Creati con successo {success_count} file MP3.")
    print(f"Cartella di destinazione: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    main()