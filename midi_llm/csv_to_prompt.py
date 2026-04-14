import csv
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Text prompt generator extracting data directly from the Groove MIDI CSV with diversified phrasing."
    )
    
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="/data/groove_info.csv", 
        help="Path to the metadata CSV file"
    )
    
    parser.add_argument(
        "--out_path", 
        type=str, 
        required=True, 
        help="Path to the output TXT file (e.g., groove_prompts.txt)"
    )
    
    parser.add_argument(
        "--limit", 
        type=int, 
        default=500, 
        help="Maximum number of prompts to generate (default: 500)"
    )

    args = parser.parse_args()

    # 2. Variable assignment and path validation
    CSV_PATH = Path(args.csv_path)
    TXT_PATH = Path(args.out_path)
    LIMIT = args.limit

    if not CSV_PATH.exists():
        print(f"Error: The CSV file does not exist at:\n{CSV_PATH}")
        return

    random.seed(42)

    print(f"--- GROOVE MIDI DIVERSIFIED PROMPT EXTRACTOR ---")
    print(f"Extracting {LIMIT} randomized prompts from file: {CSV_PATH.name}")

    try:
        TXT_PATH.parent.mkdir(parents=True, exist_ok=True)

        prompts_count = 0
        
        with open(CSV_PATH, mode='r', encoding='utf-8') as f_in:
            csv_reader = csv.DictReader(f_in)
            
            with open(TXT_PATH, "w", encoding="utf-8") as f_out:
                for row in csv_reader:
                    if prompts_count >= LIMIT:
                        break
                    
                    style = str(row.get("style", "drum")).replace('/', ' and ').strip()
                    beat_type = "drum fill" if row.get("beat_type") == 'fill' else "drum beat"
                    bpm = str(row.get("bpm", "120")).strip()
                    time_sig = str(row.get("time_signature", "4-4")).replace('-', '/').strip()
                    drummer = str(row.get("drummer", "a professional")).strip()
                    
                    templates = [
                        f"A {style} {beat_type} played in {time_sig} time at {bpm} BPM by {drummer}.",
                        f"{drummer} performs a {style} {beat_type} with a {time_sig} signature at {bpm} BPM.",
                        f"At {bpm} BPM, this is a {style} {time_sig} rhythm, specifically a {beat_type} by {drummer}.",
                        f"Capture the essence of {style} with this {beat_type} in {time_sig}. Tempo: {bpm} BPM.",
                        f"Provide a {style} drum track at {bpm} BPM with a {time_sig} time signature.",
                        f"Generate a {time_sig} {style} rhythm, playing at {bpm} beats per minute.",
                        f"Compose a {style} {beat_type} in {time_sig}. Use a tempo of {bpm} BPM. Style reference: {drummer}."
                    ]
                    prompt = random.choice(templates)
                    
                    f_out.write(f"{prompt}\n")
                    prompts_count += 1

        print(f"Operation completed successfully!")
        print(f"Created {prompts_count} diversified prompts in: {TXT_PATH.absolute()}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")

if __name__ == "__main__":
    main()