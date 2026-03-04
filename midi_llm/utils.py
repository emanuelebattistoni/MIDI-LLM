"""
Utility functions for MIDI-LLM.

This module contains helper functions for audio synthesis, MIDI conversion,
and other supporting operations. Users can safely skip this file when learning
the codebase - start with generate_vllm.py or train.py instead.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import torch

# Core dependency - required
try:
    from anticipation.convert import events_to_midi #Il programma prova ad importare la funzione event_to_midi che trasforma i numeri
    #generati dall'AI in uno spartito MIDI leggibile
except ImportError:
    print("Error: anticipation package not found. Please install it for MIDI conversion.")
    print("Install with: pip install anticipation") #Dice all'utente di installare la libreria anticipation
    sys.exit(1)

# Optional dependencies for audio synthesis
SYNTHESIS_AVAILABLE = False
LOUDNESS_NORM_AVAILABLE = False
try:
    import midi2audio   #per interfacciarsi con il fluidsynth
    import librosa      #per manipolare file audio
    import librosa.effects  
    import soundfile as sf  #per manipolare file oudio
    SYNTHESIS_AVAILABLE = True
    
    # Optional loudness normalization
    try:
        import pyloudnorm as pyln   #per la normalizzazione del volume        
        LOUDNESS_NORM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

AMT_GPT2_BOS_ID = 55026     #midi tokens
LLAMA_VOCAB_SIZE = 128256   #LLM tokens   
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"    #nome del modello

# MIDI tokens are in the extended vocabulary range
ALLOWED_TOKEN_IDS = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID))
"""Crea una lista di sicurezza, in modo che l'ia possa scegliere soltanto i token presenti in questo range e divisi in token testuale e
MIDI token"""


# ============================================================================
# Validation
# ============================================================================

def has_excessive_notes_at_any_time(
    tokens: Union[torch.Tensor, List[int]], #prende in input o tensori o liste
    max_notes_per_time: int = 64    #impostato a 64 perchè raramente un orchestra suona più di 40-50 note nello stesso identico istante
) -> bool:
    """
    Check if generated MIDI has excessive simultaneous notes at any time point.
    
    This validation helps filter out invalid or unrealistic generations that have
    too many notes playing at once, which can indicate a failure mode.
    
    Args:
        tokens: Token sequence (torch.Tensor or list of ints)
        max_notes_per_time: Maximum allowed notes at any single time point
        
    Returns:
        True if excessive notes detected, False otherwise
    """
    # Convert to tensor if needed
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens)   normalizza i dati a tensori
    
    # Extract time tokens (every 3rd token in the sequence: time, duration, note)
    times = tokens[::3] #estrae solo i token del tempo
    
    # Use torch.bincount for efficient counting
    # bincount returns counts for indices 0 to max_value
    counts = torch.bincount(times)  #Conta il numero suonato di note al secondo.
    
    # Check if any time has more than max_notes_per_time notes
    return torch.any(counts > max_notes_per_time).item()    #controlla ogni istante temporale


# ============================================================================
# Audio Synthesis
# ============================================================================

def synthesize_midi_to_audio(
    midi_path: str, 
    soundfont_path: str,
    save_mp3: bool = True,
    samplerate: Optional[int] = None,
    target_loudness: float = -18.0
) -> bool:
    """
    Synthesize MIDI file to audio (WAV/MP3) using FluidSynth with loudness normalization.
    
    Args:
        midi_path: Path to MIDI file
        soundfont_path: Path to SoundFont (.sf2) file
        save_mp3: If True, convert to MP3 and delete WAV
        samplerate: Optional sample rate for audio
        target_loudness: Target loudness in LUFS (default: -14.0, Spotify standard)
        
    Returns:
        True if successful, False otherwise
    """
    if not SYNTHESIS_AVAILABLE:
        print("Warning: Audio synthesis libraries not available. Skipping synthesis.")
        print("Install with: conda install conda-forge::fluidsynth conda-forge::ffmpeg")
        print("              pip install midi2audio librosa soundfile pyloudnorm")
        return False
    
    try:
        wav_path = midi_path.replace(".mid", ".wav")    #copia il nome del file cambiando da .mid a .wav
        
        import subprocess # Lo aggiungiamo per assicurarci che possa lanciare i comandi
        
        # Calcola il sample rate
        sr_val = str(samplerate) if samplerate is not None else "44100"
        
        # Costruisce il comando aggirando il bug di FluidSynth 2.5+
        # (Le opzioni -ni, -r, -F DEVONO stare prima dei file)
        cmd = [
            "fluidsynth", 
            "-ni", 
            "-r", sr_val, 
            "-F", wav_path, 
            soundfont_path, 
            midi_path
        ]
        
        # Lancia il comando in modo invisibile senza stampare scritte inutili
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load and trim silence from audio
        wav, sr = librosa.load(wav_path)    #Utilizza la libreria librosa per avere il wav ed il sr
        wav, _ = librosa.effects.trim(wav, top_db=30)   #Taglia via il silenzio all'inizio e alla fine del bra grazie a top_db=30
        
        # Apply loudness normalization
        if LOUDNESS_NORM_AVAILABLE:
            try:
                # Measure the loudness
                meter = pyln.Meter(sr)  #crea un fonometro virtuale basato sul sr
                loudness = meter.integrated_loudness(wav)   #analizza il brano misurando il volume medio 
                
                # Normalize to target loudness
                wav = pyln.normalize.loudness(wav, loudness, target_loudness)# Alza o abbassa il volume portandolo alla target loudness
                
                # Prevent clipping, , questa riga scala verso il basso l'intero brano quel tanto che basta per far rientrare
                # il picco più alto esattamente a 1.0
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
            except Exception as e:
                print(f"Warning: Loudness normalization failed: {e}")
        
        # Write normalized audio
        sf.write(wav_path, wav, sr)
        
        if save_mp3:
            # Convert WAV to MP3 using ffmpeg, ffmpeg è un software esterno per gestire audio e video
            mp3_path = midi_path.replace(".mid", ".mp3")    #prende il nome del file originale e ne cambia l'estensione
            if samplerate is None:
                cmd = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 {mp3_path} -y >/dev/null 2>&1"
                """prende come input il wav masterizzato, usa l'encoder lame, imposta la qualità a 2 (190-250 kbps), se il file esiste 
                già lo sovrascrive senza chiedere il permesso,>/dev/null 2>&1 Impedisce a FFmpeg di riempire il tuo terminale di scritte""" tecniche
            else:
                cmd = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2 -ar {samplerate} {mp3_path} -y >/dev/null 2>&1"
                #se è stato specificato un sr forza FFmpeg ad usare quella definizione sonora nella conversione finale    
            os.system(cmd)  #Invia il comando che abbiamo costruito sopra al sistema operativo
            
            # Remove WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path) #Rimuove il wav originale dato che è stato compresso in mp3
        
        return True
    
    except Exception as e:  #gestisce un eventuale errore
        print(f"Error synthesizing MIDI to audio: {e}")
        return False


# ============================================================================
# MIDI Generation and Saving
# ============================================================================

def save_generation(
    tokens: List[int],
    prompt: str,
    output_dir: Path,
    generation_idx: int,
    soundfont_path: Optional[str] = None,
    synthesize: bool = False,
    validate: bool = True
) -> bool:
    """
    Save generated tokens as MIDI file (and optionally audio).
    
    Args:
        tokens: List of generated token IDs (already shifted from LLAMA vocab)
        prompt: Original text prompt
        output_dir: Directory to save outputs
        generation_idx: Index of this generation (for multiple outputs)
        soundfont_path: Path to SoundFont file for synthesis
        synthesize: Whether to synthesize to audio
        validate: Whether to validate tokens before saving (checks for excessive notes)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate tokens before saving
        if validate:
            if has_excessive_notes_at_any_time(tokens, max_notes_per_time=64):#controlla se ci sono più di 64 note contemporaneamente
                print(f"  ✗ Generation {generation_idx}: Failed validation (excessive simultaneous notes)")
                return False
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompt text
        prompt_file = output_dir / "prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        # Save token sequence
        tokens_file = output_dir / f"gen_{generation_idx}_tokens.txt"
        with open(tokens_file, "w") as f:
            for token in tokens:
                f.write(f"{token}\n")
        
        # Convert tokens to MIDI
        midi_obj = events_to_midi(tokens)   #funzione che traduce i tokens generati in midi
        midi_file = output_dir / f"gen_{generation_idx}.mid"    #viene creato il nome del file .mid
        midi_obj.save(str(midi_file))       #viene salvato effettivamente il file
        
        print(f"  ✓ Saved MIDI: {midi_file}")
        
        # Optionally synthesize to audio
        if synthesize and soundfont_path:   #se presente l'opzione synthesize ed un soundfont
            success = synthesize_midi_to_audio(
                str(midi_file), #passa il midi generato
                soundfont_path,#passa il soundfont
                save_mp3=True   #richiede la compressione in mp3
            )
            if success:
                print(f"  ✓ Synthesized audio: {midi_file.with_suffix('.mp3')}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error saving generation {generation_idx}: {e}")
        return False

