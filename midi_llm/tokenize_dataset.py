"""
Groove MIDI Dataset → groove_sft_dataset_hf.jsonl
Pre-tokenized format compatible with MIDI-LLM (slseanwu/MIDI-LLM_Llama-3.2-1B)
"""

import json
import os
import tempfile
import csv
from pathlib import Path
from anticipation.convert import midi_to_events
from transformers import AutoTokenizer

# ==========================================
# MIDI-LLM CONSTANTS
# ==========================================
SYSTEM_PROMPT = "You are a world-class composer. Please compose some music according to the following description: "

LLAMA_MODEL_NAME = "slseanwu/MIDI-LLM_Llama-3.2-1B"
LLAMA_VOCAB_SIZE = 128256
AMT_GPT2_BOS_ID = 55026

OUTPUT_JSONL = "./data/groove_sft_dataset_hf.jsonl"
LOCAL_MIDI_ROOT = Path("./data/groove-midi-dataset")
SPLITS_TO_USE = {"train"} 


def build_text_prompt(row: dict) -> str:
    """Builds the text prompt from CSV metadata."""
    style = row["style"].replace("/", " ").replace("-", " ").strip()
    beat_type = "drum fill" if row["beat_type"] == "fill" else "drum beat"
    bpm = row["bpm"]
    time_sig = row["time_signature"].replace("-", "/")
    drummer = row["drummer"]

    return f"A {style} {beat_type} played in {time_sig} time at {bpm} BPM by {drummer}."


def get_midi_bytes(midi_filename: str) -> bytes | None:
    """Reads the MIDI file from local storage, returns bytes."""
    local_path = LOCAL_MIDI_ROOT / midi_filename
    if local_path.exists():
        return local_path.read_bytes()
    else:
        print(f"  [!] File not found locally: {local_path}")
        return None


def tokenize_midi_bytes(midi_bytes: bytes) -> list | None:
    """Converts MIDI bytes directly into a list of integers (AMT tokens)."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp.write(midi_bytes)
        tmp_path = tmp.name
    try:
        amt_tokens = midi_to_events(tmp_path)
        return amt_tokens
    except Exception as e:
        print(f"  [!] AMT tokenization error: {e}")
        return None
    finally:
        os.unlink(tmp_path)


def main():
    print(f"Loading Llama 3 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME)
    
    csv_path = LOCAL_MIDI_ROOT / "info.csv"
    print(f"Reading local info.csv from {csv_path}...")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["split"] in SPLITS_TO_USE]
    print(f"Rows to process (split={SPLITS_TO_USE}): {len(rows)}")

    processed = 0
    failed = 0

    # Create parent directories if they don't exist
    Path(OUTPUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        for i, row in enumerate(rows):
            midi_filename = row["midi_filename"]
            prompt_text = build_text_prompt(row)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(rows)}] {midi_filename}")

            # 1. Get raw MIDI tokens
            midi_bytes = get_midi_bytes(midi_filename)
            if midi_bytes is None:
                failed += 1
                continue

            amt_tokens = tokenize_midi_bytes(midi_bytes)
            if amt_tokens is None:
                failed += 1
                continue

            full_text_prompt = SYSTEM_PROMPT + prompt_text + " "
            
            # A. Transform the text into standard Llama 3 tokens
            text_input_ids = tokenizer(full_text_prompt, add_special_tokens=True)["input_ids"]
            
            # B. Create the special MIDI_BOS token (offset 128256)
            midi_bos_id = [AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]
            
            # C. Transform the AMT numbers into the model's extended tokens (offset 128256)
            midi_input_ids = [t + LLAMA_VOCAB_SIZE for t in amt_tokens]
            
            # D. Merge everything into a single, perfect mathematical sequence
            final_input_ids = text_input_ids + midi_bos_id + midi_input_ids

            # Save the JSON with input_ids and labels, ready for the SFTTrainer
            item = {
                "input_ids": final_input_ids,
                "labels": final_input_ids # In Causal LM, labels are identical to input_ids
            }
            
            f_out.write(json.dumps(item) + "\n")
            processed += 1

    print(f"\n--- COMPLETED ---")
    print(f"Examples written:  {processed}")
    print(f"Examples skipped:  {failed}")
    print(f"Dataset saved to:  {os.path.abspath(OUTPUT_JSONL)}")

if __name__ == "__main__":
    main()