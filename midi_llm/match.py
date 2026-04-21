import json
import re
from pathlib import Path

def main():
    BASE_DIR = Path("/home/emanuelebattistoni/Documents/workspace/MIDI-LLM")
    LISTA_ID_PATH = BASE_DIR / "assets/evaluation_set_lakh_ids.txt"
    TRAIN_JSON_PATH = BASE_DIR / "midicaps_data/train.json"
    OUTPUT_FILE = BASE_DIR / "captions_midicaps.txt"

    with open(LISTA_ID_PATH, "r") as f:
        target_ids = {line.split(']')[-1].strip() for line in f if len(line.strip()) >= 32}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as j_f:
            for line in j_f:
                try:
                    data = json.loads(line)
                    location = data.get("location", "")
                    
                    match = re.search(r'([a-f0-9]{32})', location)
                    if match:
                        file_id = match.group(1)
                        if file_id in target_ids:
                            caption = data.get("caption", "")
                            out_f.write(f"{caption}\n")
                except json.JSONDecodeError:
                    continue

if __name__ == "__main__":
    main()