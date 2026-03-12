import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

import torch
import librosa
import pathlib
import argparse

# Import tqdm for the progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Global variable to hold the CLAP model instance
_clap_model_instance = None
CLAP_AVAILABLE = False

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False

def init_clap(enable_fusion=True):
    """
    Initialize the CLAP model. 
    Feature Fusion is enabled by default to handle variable-length audio inputs.
    """
    global _clap_model_instance
    if not CLAP_AVAILABLE:
        print("Error: laion_clap or librosa libraries not found.")
        return None

    if _clap_model_instance is None:
        print(f"Initializing CLAP model (Feature Fusion: {'ENABLED' if enable_fusion else 'DISABLED'})...")
        
        # Suppress stdout during checkpoint loading to keep the console clean
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            # HTSAT is used as the default audio encoder for state-of-the-art performance.
            _clap_model_instance = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
            _clap_model_instance.load_ckpt()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        print("CLAP model loaded and ready for analysis.\n")
    
    return _clap_model_instance

def evaluate_audio_clap(text_prompt: str, audio_path: str) -> float:
    """
    Calculate the cosine similarity between the user prompt and the generated audio.
    """
    model = _clap_model_instance
    if not CLAP_AVAILABLE or model is None:
        return 0.0
    
    try:
        # Generate text embedding from the user's prompt
        text_embed = model.get_text_embedding([text_prompt])
        
        # Load audio at 48kHz sampling rate as required by the CLAP standard 
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        
        # Generate audio embedding (utilizes fusion if enabled during init)
        audio_embed = model.get_audio_embedding_from_data(x=audio_data)
        
        # Convert to Tensor and apply L2 normalization for cosine similarity calculation
        text_embed_t = torch.from_numpy(text_embed).float()
        audio_embed_t = torch.from_numpy(audio_embed).float()
        
        text_embed_t = torch.nn.functional.normalize(text_embed_t, p=2, dim=-1)
        audio_embed_t = torch.nn.functional.normalize(audio_embed_t, p=2, dim=-1)
        
        # Compute the similarity score (how well the audio matches the request) 
        similarity = torch.nn.functional.cosine_similarity(text_embed_t, audio_embed_t, dim=-1)
        
        return float(similarity.item())
        
    except Exception as e:
        if TQDM_AVAILABLE:
            tqdm.write(f"Technical error processing {audio_path}: {e}")
        else:
            print(f"Technical error processing {audio_path}: {e}")
        return -1.0

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="CLAP Audio Fidelity Analyzer.")
    parser.add_argument("--dir", type=str, required=True, help="Path to the root folder")
    parser.add_argument("--no-fusion", action="store_true", help="Disable feature fusion (use for audio < 10s)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: The path '{args.dir}' is not a valid directory.")
        sys.exit(1)

    # Initialize model with Fusion active by default for long generations
    init_clap(enable_fusion=not args.no_fusion)
    
    root_path = pathlib.Path(args.dir)
    
    # 1. Recursively find ALL .mp3 files, regardless of directory depth
    audio_files = list(root_path.rglob("*.mp3"))
    
    if not audio_files:
        print("No .mp3 files found in the specified directory tree.")
        sys.exit(0)

    results = []

    # 2. Setup the progress bar iterator over the found audio files
    if TQDM_AVAILABLE:
        iterator = tqdm(audio_files, desc="Evaluating CLAP", unit="file")
    else:
        print(f"Evaluating {len(audio_files)} files (install 'tqdm' for a progress bar)...")
        iterator = audio_files

    # 3. Process each audio file
    for audio_file in iterator:
        prompt_text = None
        
        # Look for a .txt file strictly in the SAME folder as the current .mp3
        txt_files = list(audio_file.parent.glob("*.txt"))
        
        if txt_files:
            # Assume the first .txt found contains the prompt
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
        
        # If the prompt is found, evaluate and store
        if prompt_text:
            score = evaluate_audio_clap(prompt_text, str(audio_file))
            if score >= 0:
                results.append({
                    "score": score
                })

    # 4. Handle edge case where files were found but no valid pairs were evaluated
    if not results:
        print("\nNo valid audio-prompt pairs were successfully evaluated.")
        return

    # 5. Calculate Average 
    avg_score = sum(res["score"] for res in results) / len(results)

    # 6. Print ONLY the final summary
    print("\n" + "=" * 50)
    print(f"FINAL EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total files analyzed : {len(results)}")
    print(f"AVERAGE CLAP SCORE   : {avg_score:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()