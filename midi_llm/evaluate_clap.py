#!/usr/bin/env python3
import os
import sys
import torch

# Global variable to hold the model instance
_clap_model_instance = None
CLAP_AVAILABLE = False

try:
    import laion_clap
    import librosa
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False

def is_clap_available():
    return CLAP_AVAILABLE

def init_clap():
    """Load the CLAP model into memory silently."""
    global _clap_model_instance
    if not CLAP_AVAILABLE:
        print("Warning: laion_clap or librosa not installed. CLAP evaluation disabled.")
        return None

    if _clap_model_instance is None:
        print("\nLoading CLAP model...")
        # Silence the output during loading
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            _clap_model_instance = laion_clap.CLAP_Module(enable_fusion=False)
            _clap_model_instance.load_ckpt()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        print("CLAP model loaded successfully.\n")
    
    return _clap_model_instance

def evaluate_audio_clap(text_prompt: str, audio_path: str) -> float:
    """Calculate the cosine similarity score between text and audio."""
    global _clap_model_instance
    
    if not CLAP_AVAILABLE or _clap_model_instance is None:
        return 0.0
    
    try:
        # Extract text embedding
        text_embed = _clap_model_instance.get_text_embedding([text_prompt])
        
        # Load audio at 48kHz (CLAP standard)
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        
        # Extract audio embedding
        audio_embed = _clap_model_instance.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
        
        # Calculate cosine similarity using Torch
        text_embed_t = torch.tensor(text_embed)
        audio_embed_t = torch.tensor(audio_embed)
        similarity = torch.nn.functional.cosine_similarity(text_embed_t, audio_embed_t)
        
        return float(similarity.item())
    except Exception as e:
        print(f"Error during CLAP evaluation for {audio_path}: {e}")
        return -1.0