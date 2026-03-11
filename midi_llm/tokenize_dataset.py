import json
import tempfile
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from anticipation.convert import midi_to_events

def main():
    # Output file path for the fine-tuning dataset
    OUTPUT_JSONL = "groove_sft_dataset_hf.jsonl"
    
    # 1. Initialize Llama 3.2 1B Tokenizer
    print("Loading Llama 3.2 1B Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("slseanwu/MIDI-LLM_Llama-3.2-1B")
    
    # 2. Download and load the Groove MIDI dataset from Hugging Face Hub
    print("Downloading/Loading Groove MIDI dataset from Hugging Face...")
    # This command downloads the dataset on the first execution and utilizes 
    # the local cache for subsequent runs.
    dataset = load_dataset("schismaudio/groove-midi-dataset")
    
    processed_files = 0
    failed_files = 0

    print("Processing online dataset and tokenizing MIDI files...")

    # Open the output file in write mode for streaming
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f_out:
        
        # Iterate through available dataset splits (e.g., train, validation, test)
        for split in dataset.keys():
            print(f"Processing split: {split}...")
            
            for row in dataset[split]:
                # 3. Metadata extraction for text prompt construction
                style = str(row.get("style", "drum")).replace('/', ' and ').strip()
                beat_type = "drum fill" if row.get("beat_type") == 'fill' else "drum beat"
                bpm = str(row.get("bpm", "120")).strip()
                time_sig = str(row.get("time_signature", "4-4")).strip()
                
                text_prompt = f"A {style} {beat_type} played in {time_sig} time at {bpm} BPM."
                
                # 4. Safe MIDI file extraction
                # Hugging Face provides MIDI data either as raw bytes or cached file paths.
                # To ensure compatibility with the AMT library (which requires a file path),
                # we write bytes to a temporary file, process it, and delete it immediately after.
                try:
                    # Create a secure temporary file
                    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi:
                        temp_midi_path = temp_midi.name
                        
                        # Handle cases where MIDI is provided as raw bytes within a dictionary
                        if "midi" in row and isinstance(row["midi"], dict) and "bytes" in row["midi"]:
                            temp_midi.write(row["midi"]["bytes"])
                        else:
                            # Fallback to direct file path if provided by the dataset structure
                            temp_midi_path = row.get("midi_filename", temp_midi_path)
                    
                    # Perform AMT (Anticipatory Music Transformer) tokenization
                    amt_tokens = midi_to_events(temp_midi_path)
                    midi_string = " ".join([str(token) for token in amt_tokens])
                    
                except Exception as e:
                    print(f"Error processing MIDI structure: {e}")
                    failed_files += 1
                    continue
                finally:
                    # Cleanup: Remove the temporary file if it was created during this iteration
                    if os.path.exists(temp_midi_path) and temp_midi_path.startswith(tempfile.gettempdir()):
                        os.remove(temp_midi_path)
                
                # 5. Build conversation structure for Llama 3
                conversation = [
                    {"role": "user", "content": text_prompt},
                    {"role": "assistant", "content": midi_string}
                ]
                
                # 6. Apply Llama 3 chat template for instruction fine-tuning
                final_text = tokenizer.apply_chat_template(conversation, tokenize=False)
                
                # 7. Write processed sample to JSONL file
                item = {"text": final_text}
                f_out.write(json.dumps(item) + "\n")
                
                processed_files += 1
                if processed_files % 100 == 0:
                    print(f"Tokenized {processed_files} files from Hugging Face...")

    # Final execution report
    print(f"\n--- COMPLETED ---")
    print(f"MIDI files successfully tokenized from cloud: {processed_files}")
    if failed_files > 0:
        print(f"Skipped files: {failed_files}")
    print(f"Dataset saved to: {os.path.abspath(OUTPUT_JSONL)}")

if __name__ == "__main__":
    main()