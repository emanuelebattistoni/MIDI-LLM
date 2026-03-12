import json
import argparse
import sys
from pathlib import Path

def calculate_clap_average(json_path):
    """
    Parses a MIDI-LLM generation stats JSON file and calculates the 
    average CLAP similarity score.
    """
    path = Path(json_path)
    
    if not path.exists():
        print(f"Error: File '{json_path}' not found.")
        return

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Access the clap_scores dictionary
        clap_scores_dict = data.get("clap_scores", {})
        
        if not clap_scores_dict:
            print(f"No CLAP scores found in '{json_path}'.")
            return

        # Extract values
        raw_scores = list(clap_scores_dict.values())
        
        # Filtering: We exclude -1.0 as it represents a technical processing error 
        # encountered during the evaluation script execution.
        valid_scores = [s for s in raw_scores if s != -1.0]
        error_count = len(raw_scores) - len(valid_scores)

        if not valid_scores:
            print("No valid scores available to calculate an average.")
            return

        # Statistical calculation
        average_score = sum(valid_scores) / len(valid_scores)
        
        # Display Results
        print(f"\n{'='*40}")
        print(f" CLAP STATISTICS: {path.name}")
        print(f"{'='*40}")
        print(f" Total files evaluated:  {len(raw_scores)}")
        print(f" Technical errors (-1):  {error_count}")
        print(f" Valid scores used:      {len(valid_scores)}")
        print(f"{'-'*40}")
        print(f" AVERAGE CLAP SCORE:     {average_score:.4f}")
        print(f"{'='*40}\n")

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{json_path}'. Check file format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and average CLAP scores from a MIDI-LLM stats JSON file."
    )
    
    parser.add_argument(
        "input_json", 
        type=str, 
        help="Path to the generation_stats.json file"
    )

    args = parser.parse_args()
    calculate_clap_average(args.input_json)

if __name__ == "__main__":
    main()