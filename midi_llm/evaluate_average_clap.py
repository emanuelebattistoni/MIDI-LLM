import os
import sys
import warnings
import logging
import torch
import librosa
import pathlib
import argparse

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)

_clap_model_instance = None

def init_clap():
    global _clap_model_instance
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        import laion_clap

        _clap_model_instance = laion_clap.CLAP_Module(enable_fusion=True)
        _clap_model_instance.load_ckpt()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    return _clap_model_instance

def evaluate_audio_clap(text_prompt, audio_path):
    try:
        text_embed = _clap_model_instance.get_text_embedding([text_prompt])
        audio_data, _ = librosa.load(audio_path, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        audio_embed = _clap_model_instance.get_audio_embedding_from_data(x=audio_data)
        
        t_embed = torch.from_numpy(text_embed).float()
        a_embed = torch.from_numpy(audio_embed).float()
        
        t_embed = torch.nn.functional.normalize(t_embed, p=2, dim=-1)
        a_embed = torch.nn.functional.normalize(a_embed, p=2, dim=-1)
        
        sim = torch.nn.functional.cosine_similarity(t_embed, a_embed, dim=-1)
        return float(sim.item())
    except Exception as e:
        print(f"Errore interno CLAP: {e}")
        return -1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()
    
    init_clap()
    
    root = pathlib.Path(args.dir)
    files = sorted(list(root.rglob("*.mp3")))
    results = []
    
    for f_path in files:
        txt_list = sorted(list(f_path.parent.glob("*.txt")))
        prompt_file = None
        
        if len(txt_list) >= 2:
            prompt_file = txt_list[1] 
        else:
            print(f"[SKIP] {f_path.name}: Non ci sono almeno 2 file .txt in questa cartella.")
            continue
            
        if prompt_file:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
                
                score = evaluate_audio_clap(prompt, str(f_path))
                
                if score >= -0.9:
                    results.append(score)
                    print(f"{f_path.name} (valutato su {prompt_file.name}): {score:.4f}")
            except Exception as e:
                print(f"[ERROR] Impossibile valutare {f_path.name}: {e}")
                continue
                
    if results:
        media = sum(results) / len(results)
        print(f"\n{'='*50}")
        print(f"Analizzati: {len(results)} file")
        print(f"MEDIA CLAP: {media:.4f}")
        print(f"{'='*50}")
    else:
        print("\nNessun file è stato valutato con successo.")

if __name__ == "__main__":
    main()