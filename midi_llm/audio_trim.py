import os
from pathlib import Path
from pydub import AudioSegment, effects
from tqdm import tqdm
# Set the source and destination paths for the reference and evaluation datasets
SOURCE_DIR = Path("./data/maestro_eval_set")
DEST_DIR = Path("./data/maestro_eval_final")

# Processing parameters
MAX_SECONDS = 30  # Maximum length for trimming
MIN_SECONDS = 29  # Minimum duration threshold for filtering
TARGET_DBFS = -16.0  # Loudness normalization target
# Note: TARGET_SAMPLE_RATE removed to maintain original sample rate (44.1 kHz)


def main():
    """
    Standardize audio files by filtering duration, converting to mono,
    trimming to a fixed length, and normalizing loudness.
    """
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory {SOURCE_DIR} does not exist.")
        return

    # Create destination directory if it doesn't exist
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Collect all supported audio files
    audio_files = [f for f in SOURCE_DIR.iterdir() if f.suffix.lower() in ['.mp3', '.wav']]
    
    if not audio_files:
        print(f"No audio files found in {SOURCE_DIR}")
        return

    print(f"Processing: Filtering > {MIN_SECONDS}s | Mono Conversion | Trimming ({MAX_SECONDS}s) | Normalization...")

    skipped_count = 0  # Initialize the counter for discarded files

    for f in tqdm(audio_files):
        try:
            # 1. Load audio file
            audio = AudioSegment.from_file(f)
            
            # 2. Duration Control
            # Pydub measures duration in milliseconds (e.g., 29s = 29000ms)
            if len(audio) < MIN_SECONDS * 1000:
                skipped_count += 1
                continue  # Skip files that do not meet the minimum duration requirement
            
            # 3. Channel Conversion
            # Convert to Mono, maintaining the original sample rate (e.g., 44.1kHz)
            audio = audio.set_channels(1)
            
            # 4. Trimming
            # Cut the audio to the specified maximum length
            trimmed = audio[:MAX_SECONDS * 1000]
            
            # 5. Normalization (Peak + Targeted Loudness)
            normalized = effects.normalize(trimmed)
            
            # Avoid processing silent/invalid audio
            if normalized.dBFS == float('-inf'):
                continue
                
            # Apply gain to reach the exact target dBFS
            change_in_db = TARGET_DBFS - normalized.dBFS
            final_audio = normalized.apply_gain(change_in_db)
            
            # 6. Export as WAV
            output_path = DEST_DIR / f.with_suffix('.wav').name
            final_audio.export(output_path, format="wav")
            
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Summary report
    print(f"\nTask complete! Dataset ready at: {DEST_DIR.absolute()}")
    
    if skipped_count > 0:
        print(f"Files discarded (duration < {MIN_SECONDS}s): {skipped_count}")
        print(f"Files successfully processed: {len(audio_files) - skipped_count}")

if __name__ == "__main__":
    main()