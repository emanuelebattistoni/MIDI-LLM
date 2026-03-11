"""
Utility functions for MIDI-LLM.

This module contains helper functions for audio synthesis, MIDI conversion,
and other supporting operations. Users can safely skip this file when learning
the codebase - start with generate_transformers.py or train_lora.py instead.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import torch

# Core dependency for MIDI token conversion
try:
    # Converts AI-generated token sequences back into readable MIDI objects
    from anticipation.convert import events_to_midi 
except ImportError:
    print("Error: 'anticipation' package not found. MIDI conversion will fail.")
    print("Please install it using: pip install anticipation")
    sys.exit(1)

# Optional dependencies for high-quality audio synthesis
SYNTHESIS_AVAILABLE = False
LOUDNESS_NORM_AVAILABLE = False
try:
    import librosa         # Audio analysis and processing
    import librosa.effects 
    import soundfile as sf  # Audio file writing
    SYNTHESIS_AVAILABLE = True
    
    # Optional library for LUFS loudness normalization
    try:
        import pyloudnorm as pyln       
        LOUDNESS_NORM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


# ============================================================================
# Constants
# ============================================================================

AMT_GPT2_BOS_ID = 55026      # Beginning of Sequence ID for MIDI tokens
LLAMA_VOCAB_SIZE = 128256    # Standard Llama 3.2 vocabulary offset
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-1B"

# MIDI tokens are stored in the extended vocabulary range above standard text tokens
ALLOWED_TOKEN_IDS = list(range(LLAMA_VOCAB_SIZE, LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID))


# ============================================================================
# Validation
# ============================================================================

def has_excessive_notes_at_any_time(
    tokens: Union[torch.Tensor, List[int]], 
    max_notes_per_time: int = 64
) -> bool:
    """
    Check if the generated MIDI contains an unrealistic number of simultaneous notes.
    
    This validation filters out 'hallucinated' sequences where the model generates
    excessive polyphony, which is often a sign of generation failure.
    
    Args:
        tokens: Sequence of token IDs (Tensor or List)
        max_notes_per_time: Threshold for maximum simultaneous notes (default: 64)
        
    Returns:
        True if the sequence exceeds the threshold, False otherwise.
    """
    # Normalize input to a PyTorch tensor
    if isinstance(tokens, list):
        tokens = torch.tensor(tokens)
    
    # Extract 'onset time' tokens (AMT format uses triplets: time, duration, pitch)
    times = tokens[::3]
    
    # Efficiently count occurrences of each time index using bincount
    counts = torch.bincount(times)
    
    # Identify if any specific time point exceeds the polyphony limit
    return torch.any(counts > max_notes_per_time).item()


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
    Synthesize MIDI to audio (WAV/MP3) using FluidSynth with loudness normalization.
    
    Args:
        midi_path: Path to the source MIDI file.
        soundfont_path: Path to the .sf2 SoundFont file.
        save_mp3: If True, encodes to MP3 and removes the intermediate WAV.
        samplerate: Audio sampling rate (default: 44100).
        target_loudness: Target Integrated Loudness in LUFS.
        
    Returns:
        True if synthesis and processing were successful.
    """
    if not SYNTHESIS_AVAILABLE:
        print("Warning: Audio synthesis libraries missing. Skipping audio generation.")
        print("System requirements: fluidsynth, ffmpeg")
        print("Python requirements: librosa, soundfile, pyloudnorm")
        return False
    
    try:
        wav_path = midi_path.replace(".mid", ".wav")
        sr_val = str(samplerate) if samplerate is not None else "44100"
        
        # FluidSynth Command Construction:
        # Note: In FluidSynth 2.5+, the -F (file output) flag and SoundFont path 
        # must precede the MIDI file to avoid parsing errors.
        cmd = [
            "fluidsynth", 
            "-ni",                # Non-interactive mode
            "-r", sr_val,         # Set audio sample rate
            "-F", wav_path,       # Render output directly to WAV file
            soundfont_path,      
            midi_path
        ]
        
        # Execute synthesis silently via subprocess
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Load audio and perform aggressive silence trimming (-30dB threshold)
        wav, sr = librosa.load(wav_path, sr=int(sr_val))
        wav, _ = librosa.effects.trim(wav, top_db=30)
        
        # Apply LUFS Loudness Normalization
        if LOUDNESS_NORM_AVAILABLE:
            try:
                meter = pyln.Meter(sr)  # Initialize virtual loudness meter
                loudness = meter.integrated_loudness(wav)
                
                # Adjust volume to match the target loudness profile
                wav = pyln.normalize.loudness(wav, loudness, target_loudness)
                
                # Peak Normalization: Prevent digital clipping by scaling the signal
                # so the absolute peak resides exactly at 1.0 (0 dBFS).
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / max(abs(wav.max()), abs(wav.min()))
            except Exception as e:
                print(f"Warning: Loudness normalization failed: {e}")
        
        # Save the finalized WAV file
        sf.write(wav_path, wav, sr)
        
        if save_mp3:
            # Convert WAV to MP3 using FFmpeg with high-quality LAME encoding
            mp3_path = midi_path.replace(".mid", ".mp3")
            
            # -qscale:a 2 provides a variable bitrate (~190-250 kbps)
            # >/dev/null 2>&1 silences technical FFmpeg logs
            ffmpeg_base = f"ffmpeg -i {wav_path} -codec:a libmp3lame -qscale:a 2"
            if samplerate:
                cmd = f"{ffmpeg_base} -ar {samplerate} {mp3_path} -y >/dev/null 2>&1"
            else:
                cmd = f"{ffmpeg_base} {mp3_path} -y >/dev/null 2>&1"
                
            os.system(cmd)
            
            # Clean up temporary WAV file
            if os.path.exists(wav_path):
                os.remove(wav_path)
        
        return True
    
    except Exception as e:
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
    Save generated token sequences as MIDI files and optional audio renders.
    
    Args:
        tokens: MIDI token IDs (integer list).
        prompt: The input text description.
        output_dir: Destination folder path.
        generation_idx: Sequential ID for the generation.
        soundfont_path: Path to the .sf2 instrument file.
        synthesize: If True, triggers audio synthesis.
        validate: If True, performs polyphony safety checks.
        
    Returns:
        True if all files were successfully saved.
    """
    try:
        # Perform structural validation
        if validate:
            if has_excessive_notes_at_any_time(tokens, max_notes_per_time=64):
                print(f"  ✗ Generation {generation_idx}: Failed validation (excessive polyphony)")
                return False
        
        # Create destination directory tree
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata Persistence: Save the original prompt for reference
        prompt_file = output_dir / "prompt.txt"
        with open(prompt_file, "w") as f:
            f.write(prompt)
        
        # Debugging Persistence: Save the raw token IDs
        tokens_file = output_dir / f"gen_{generation_idx}_tokens.txt"
        with open(tokens_file, "w") as f:
            for token in tokens:
                f.write(f"{token}\n")
        
        # MIDI Conversion: Translate numerical tokens into a standard MIDI object
        midi_obj = events_to_midi(tokens)
        midi_file = output_dir / f"gen_{generation_idx}.mid"
        midi_obj.save(str(midi_file))
        
        print(f"  ✓ Saved MIDI: {midi_file}")
        
        # Optional Audio Synthesis Pipeline
        if synthesize and soundfont_path:
            success = synthesize_midi_to_audio(
                str(midi_file),
                soundfont_path,
                save_mp3=True
            )
            if success:
                print(f"  ✓ Synthesized audio: {midi_file.with_suffix('.mp3')}")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error saving generation {generation_idx}: {e}")
        return False