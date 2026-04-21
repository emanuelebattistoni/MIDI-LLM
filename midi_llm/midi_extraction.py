import os
import subprocess
import shutil
from pathlib import Path
import tqdm

def check_dependencies():
    for tool in ["fluidsynth", "ffmpeg"]:
        if shutil.which(tool) is None:
            print(f"ERROR: {tool} non è installato.")
            exit(1)

def main():
    BASE_DIR = Path("/home/emanuelebattistoni/Documents/workspace/MIDI-LLM")
    LMD_SOURCE_DIR = BASE_DIR / "lmd_full" 
    LISTA_ID_PATH = BASE_DIR / "./assets/evaluation_set_lakh_ids.txt"
    OUTPUT_DIR = BASE_DIR / "lakh_synthesis_results"
    SOUNDFONT_PATH = BASE_DIR / "soundfonts/FluidR3_GM/FluidR3_GM.sf2"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_dependencies()

    if not LISTA_ID_PATH.exists():
        print(f"Errore: {LISTA_ID_PATH} non trovato.")
        return
    
    ids = []
    with open(LISTA_ID_PATH, "r") as f:
        for line in f:
            clean_id = line.split(']')[-1].strip()
            if len(clean_id) >= 32:
                ids.append(clean_id)

    success_count = 0

    for codice in tqdm.tqdm(ids, desc="Sintesi"):
        # Struttura LMD: sottocartella basata sul primo carattere dell'hash
        char = codice[0]
        # Percorso completo del file MIDI
        midi_path = LMD_SOURCE_DIR / char / f"{codice}.mid"
        
        if not midi_path.exists():
            continue

        temp_wav = OUTPUT_DIR / f"{codice}.wav"
        final_mp3 = OUTPUT_DIR / f"{codice}.mp3"

        try:
            # Sintesi MIDI -> WAV usando FluidSynth
            subprocess.run([
                "fluidsynth", "-ni", "-a", "null", "-F", str(temp_wav), 
                "-r", "44100", str(SOUNDFONT_PATH), str(midi_path)
            ], capture_output=True, check=True)

            # Conversione WAV -> MP3 usando FFmpeg
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(temp_wav), 
                "-q:a", "2", "-map", "a", str(final_mp3)
            ], capture_output=True, check=True)

            success_count += 1
        except Exception:
            pass
        finally:
            if temp_wav.exists():
                temp_wav.unlink()

    print(f"Operazione completata. Creati {success_count} file MP3 su {len(ids)} ID processati.")

if __name__ == "__main__":
    main()