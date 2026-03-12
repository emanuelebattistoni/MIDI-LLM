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
            _clap_model_instance = laion_clap.CLAP_Module(enable_fusion=True)
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
        # 1. Omit use_tensor=True to allow the library to return its default NumPy arrays.
        # This prevents internal type conflicts within the laion_clap library.
        text_embed = _clap_model_instance.get_text_embedding([text_prompt])
        
        # Load the audio file at 48kHz (CLAP standard requirement)
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        
        # 2. Omit use_tensor=True here as well for consistency.
        audio_embed = _clap_model_instance.get_audio_embedding_from_data(x=audio_data)
        
        # SAFE TENSOR CONVERSION
        # Convert the raw NumPy arrays output by the library explicitly 
        # into PyTorch Tensors (float32) to ensure safe mathematical operations.
        text_embed_t = torch.from_numpy(text_embed).float()
        audio_embed_t = torch.from_numpy(audio_embed).float()
        
        # Normalize the vectors (L2 norm) along the last dimension
        text_embed_t = torch.nn.functional.normalize(text_embed_t, p=2, dim=-1)
        audio_embed_t = torch.nn.functional.normalize(audio_embed_t, p=2, dim=-1)
        
        # Compute the cosine similarity
        similarity = torch.nn.functional.cosine_similarity(text_embed_t, audio_embed_t, dim=-1)
        
        return float(similarity.item())
        
    except Exception as e:
        print(f"Error during CLAP evaluation for {audio_path}: {e}")
        return -1.0