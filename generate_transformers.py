#!/usr/bin/env python3
"""
MIDI-LLM: Text-to-MIDI Generation using HuggingFace Transformers

This script generates MIDI files from text prompts using the MIDI-LLM model with HuggingFace backend.
Simpler to set up than vLLM but slower for inference.
"""
import os
import sys
import warnings
import logging
from transformers import logging as hf_logging
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()

import json # Used to read and write data in JSON format
import time # Manages time and benchmarking
import argparse # Used to create command-line interfaces (CLI)
from pathlib import Path # Modern, object-oriented way to handle file and directory paths
from datetime import datetime # Used to get the current date and time
from typing import List, Optional # Type hints to clarify expected data types; Optional means a parameter can be None

import torch # Main framework used for tensor computation and deep learning
import tqdm # Purely visual library for progress bars
from transformers import AutoTokenizer, AutoModelForCausalLM # Main classes from the Hugging Face library
from transformers import StoppingCriteria, StoppingCriteriaList    # Added to handle the custom progress bar

# AutoTokenizer: Translates human text into a sequence of numbers (tokens)
# AutoModelForCausalLM: Loads the actual weights of the model

# Import helper functions and constant
from midi_llm.utils import (
    save_generation,    # Save generated tokens as MIDI file
    AMT_GPT2_BOS_ID,    # 55026
    LLAMA_VOCAB_SIZE,   # 128256
    LLAMA_MODEL_NAME,   # meta-llama/Llama-3.2-1B
)

import mido

# Import CLAP evaluation module 
from midi_llm.evaluate_clap import init_clap, evaluate_audio_clap, is_clap_available

from peft import PeftModel

# Standard General MIDI Map
GM_INSTRUMENTS = {
    # --- Piano (0-7) ---
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet",
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone",
    12: "Marimba", 13: "Xylophone", 14: "Tubular Bells", 15: "Dulcimer",
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ", 19: "Church Organ",
    20: "Reed Organ", 21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)", 29: "Overdriven Guitar",
    30: "Distortion Guitar", 31: "Guitar Harmonics",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    35: "Fretless Bass", 36: "Slap Bass 1", 37: "Slap Bass 2",
    38: "Synth Bass 1", 39: "Synth Bass 2",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass",
    44: "Tremolo Strings", 45: "Pizzicato Strings", 46: "Orchestral Harp", 47: "Timpani",
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "SynthStrings 1",
    51: "SynthStrings 2", 52: "Choir Aahs", 53: "Voice Oohs",
    54: "Synth Voice", 55: "Orchestra Hit",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 62: "SynthBrass 1", 63: "SynthBrass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)", 84: "Lead 5 (charang)", 85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)",
    91: "Pad 4 (choir)", 92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)",
    94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)",
    99: "FX 4 (atmosphere)", 100: "FX 5 (brightness)", 101: "FX 6 (goblins)",
    102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto",
    108: "Kalimba", 109: "Bagpipe", 110: "Fiddle", 111: "Shanai",
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock",
    116: "Taiko Drum", 117: "Melodic Tom", 118: "Synth Drum", 119: "Reverse Cymbal",
    120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore",
    123: "Bird Tweet", 124: "Telephone Ring", 125: "Helicopter",
    126: "Applause", 127: "Gunshot"
}

def get_instruments_from_midi_file(midi_path: str) -> List[str]:
    """Extract instrument names, including implicit Drum Kits on Channel 10."""
    try:
        mid = mido.MidiFile(midi_path)
        found_instruments = set()
        has_drums = False
        
        for track in mid.tracks:
            for msg in track:
                # 1. Look for standard program changes (ignoring the drum channel)
                if msg.type == 'program_change' and msg.channel != 9:
                    name = GM_INSTRUMENTS.get(msg.program, f"Unknown ({msg.program})")
                    found_instruments.add(name)
                
                # 2. Check if there are notes played on Channel 10 (index 9)
                elif msg.type in ['note_on', 'note_off'] and msg.channel == 9:
                    has_drums = True

        # If notes were detected on channel 10, add the Drum Kit
        if has_drums:
            found_instruments.add("Drum Kit (Channel 10)")
            
        return list(found_instruments) if found_instruments else [GM_INSTRUMENTS[0]]
    except Exception as e:
        print(f"Error analyzing MIDI instruments: {e}")
        return [GM_INSTRUMENTS[0]]
    
    # Return Grand Piano if the list is empty (MIDI default)
    return list(found_instruments) if found_instruments else [GM_INSTRUMENTS[0]]

class ProgressMonitor(StoppingCriteria):
    def __init__(self, max_new_tokens, prompt_length):
        self.prompt_length = prompt_length
        # Create the progress bar with a purely graphical and speed-oriented format
        self.pbar = tqdm.tqdm(
            total=max_new_tokens, 
            desc="Generating:", 
            position=1, 
            leave=False,
            # {bar} creates the visual blocks, we removed {n_fmt} (the raw numbers)
            bar_format="{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Calculate how many tokens (notes) have been generated
        current_tokens = input_ids.shape[1] - self.prompt_length
        increment = current_tokens - self.pbar.n
        
        # Advance the visual progress bar
        if increment > 0:
            self.pbar.update(increment)
            
        return False 
        
    def close(self):
        self.pbar.close()

# Default generation parameters
DEFAULT_TEMPERATURE = 1.0   # Probabilities "flatten" out. The model gives a chance to less obvious notes
DEFAULT_TOP_P = 0.98        # Nucleus Sampling, discards the top 2% most absurd and improbable notes
DEFAULT_MAX_TOKENS = 2046   # Maximum length of the generated composition
DEFAULT_N_OUTPUTS = 4       # Give more outputs for variability

# Model Loading
def prepare_hf_model(model_path: str, lora_path: str = None):
    """
    Initialize HuggingFace model in BFloat16.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        model
    """
    # Print a visual log
    print(f"\n{'='*70}")
    print("Model Configuration")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Precision: BFloat16")
    print(f"{'='*70}\n")
    
    # Load model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,             # Path to model checkpoint
        dtype=torch.bfloat16,   # DataType of the model parameters
        trust_remote_code=True  # Allow the computer to execute the remote code attached to the model
    ).to(device="cuda")         # Transfers the entire model to the NVIDIA GPU's VRAM
    
    if lora_path:
        print(f"Applicazione e fusione adattatore LoRA al 30% da: {lora_path}")
        model = PeftModel.from_pretrained(
            model, 
            lora_path,
            adapter_name="lora_100"
        )
         
        model.add_weighted_adapter(
            adapters=["lora_100"], 
            weights=[0.3], 
            adapter_name="lora_30"
        )
        model.set_adapter("lora_30")
        
        model = model.merge_and_unload()
    
    model.eval()
    print(f" Model loaded and merged successfully\n")
    
    return model


def generate_from_prompts_hf(
    # Takes as input:
    model,                      # The loaded model
    tokenizer: AutoTokenizer,   # The tokenizer of the loaded model
    prompts: List[str],         # List of text prompts
    output_dir: Path,           # Path of the main folder where subfolders will be created
    model_path: str,            # Path of the model being loaded
    soundfont_path: Optional[str] = None,
    synthesize: bool = False,   # If false, the function will not try to convert the MIDI file to mp3
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_outputs: int = DEFAULT_N_OUTPUTS,
    system_prompt: Optional[str] = None,
    use_clap: bool = False
     # Since it is None, the code will use the default one
) -> dict:
    """
    Generate MIDI from text prompts using HuggingFace model.
    """
    # Default system prompt --> update this if re-training with a different prefix
    if system_prompt is None:
        system_prompt = "You are a world-class composer. Please compose some music according to the following description: "
    
    stats = {
        "total_prompts": len(prompts),  # Counts the number of provided prompts
        "successful_generations": 0,    # Initializes successful generations counter
        "failed_generations": 0,        # Initializes failed generations counter
        "generation_times": [],         # Creates an empty list to save the time taken for generations
        "output_files": [],             # Creates an empty list to save the exact file paths
        "midi_instruments": {} ,        # Dictionary for the extracted used instruments
        "clap_scores" :{}
    }
    
    """
    Starts the loop:
    - tqdm(prompts): Generates a progress bar based on the length of the prompts list.
    - enumerate: Associates an index to each prompt in the list starting from zero.
    """
    
    for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Generating")):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")   # Prints current progress and prompt content
        
        # Create output directory for this prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # Saves the date and time the prompt was received
        prompt_output_dir = output_dir / f"{timestamp}_prompt_{idx+1}" # Creates the specific directory path
        
        # Prepare full prompt, add space to the end of each prompt to match training
        full_prompt = system_prompt + prompt + " "
            
        # Tokenize
        
        """
        The tokenizer takes the prompt as input, tokenizes it according to the model's vocabulary, 
        and returns a PyTorch tensor without padding. input_ids selects the actual numerical sequence.
        """
        
        llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
        input_ids = llama_input["input_ids"]
        
        # Add MIDI BOS token
        """
        The MIDI Beginning of Sequence token is calculated and transformed into a tensor so that it 
        can be concatenated to the text tensor. This forces the model to generate musical notes.
        dim=1 ensures it is concatenated along the sequence dimension (columns).
        """
        
        midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]]) # Two [] for dimensions (Batch, Sequence)
        input_ids = torch.cat([input_ids, midi_bos], dim=1) # Concatenate by column
        
        # Move to device
        """
        model.parameters() returns a generator containing all parameters.
        next() extracts the first element.
        .device checks if that component is on CPU or GPU.
        .to(device) moves our input data to the same hardware device.
        """
       
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)    # Move the data to the device where the model is loaded
        
        # Generate multiple outputs
        start_time = time.time()    # Start the stopwatch for benchmarking
        
        # 1. Measure how many tokens our textual request occupies
        prompt_len = input_ids.shape[1]
        
        # 2. Pass BOTH values to the monitor (maximum limit and initial length)
        monitor = ProgressMonitor(max_tokens, prompt_len) # Initialize the monitor

        with torch.no_grad():       # Switch to inference mode, avoiding saving intermediate gradient steps
            outputs = model.generate(
                input_ids=input_ids,    # Pass the tensor built so far to the model
                do_sample=True,         # If true, adds variability; false returns only the absolute most probable note
                max_new_tokens=max_tokens,  
                temperature=temperature,    
                top_p=top_p,
                num_return_sequences=n_outputs,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([monitor])  # Added for the progress bar 
            )

        monitor.close() # Closes the secondary progress bar when finished
        
        generation_time = time.time() - start_time  # Calculates the time spent generating
        
        if idx > 0:  # Skip first generation for timing (warmup), to avoid measuring model loading/compilation time
            stats["generation_times"].append(generation_time) # Adds the generation time for each prompt to the list
        
        print(f"Generation time: {generation_time:.2f}s") # Prints the generation time
        
        # Extract only the generated tokens (remove prompt)
        prompt_len = input_ids.shape[1]     # Measures the length (in tokens) of the initial request
        outputs = outputs[:, prompt_len:]   # Considers only the tokens generated by the AI
        
        # Shift tokens back to MIDI vocab range
        outputs = outputs - LLAMA_VOCAB_SIZE    # Output is shifted back to the MIDI range so notes are recognized
        outputs = outputs.cpu().tolist()        # Data is moved back to CPU and converted to a standard Python list
        
        # Save all outputs for this prompt
        successful_outputs = 0      # Initializes successful outputs counter to 0
        prompt_files = []          # Initializes a list to save the paths of everything generated for this prompt
        
        
        # outputs is the list of all melodies generated by the AI
        for output_idx, midi_tokens in enumerate(outputs):
            # Save generation, Save generated tokens as MIDI file
            success = save_generation(
               # Inputs 
                tokens=midi_tokens,     # List of clean numbers (notes) built previously
                prompt=prompt,          # Original textual description
                output_dir=prompt_output_dir,   # Specific folder where the output should be saved
                generation_idx=output_idx + 1,  # Used to number the file sequence
                soundfont_path=soundfont_path,  # Path to the .sf2 file for the "virtual musical instrument"
                synthesize=synthesize           # Whether to synthesize the file to mp3 or not
            )
            
            if success:
                successful_outputs += 1
                midi_file = prompt_output_dir / f"gen_{output_idx + 1}.mid"
                prompt_files.append(str(midi_file))
                
                rel_path = str(midi_file.relative_to(output_dir))
                stats["midi_instruments"][rel_path] = get_instruments_from_midi_file(str(midi_file))
                    
                if synthesize and soundfont_path:                           # If option is active, the .mp3 file name is constructed
                    mp3_file = prompt_output_dir / f"gen_{output_idx + 1}.mp3"
                    if mp3_file.exists():
                        prompt_files.append(str(mp3_file))                  # Added to the list of files

        if  use_clap and is_clap_available() and synthesize and successful_outputs > 0:
            print(f"\n CLAP is evaluating {successful_outputs} tracks...")
            for f_path in prompt_files:
                if f_path.endswith('.mp3'):
                    score = evaluate_audio_clap(prompt, f_path)
                    rel_mp3_path = str(Path(f_path).relative_to(output_dir))
                    stats["clap_scores"][rel_mp3_path] = round(score, 4)
                    print(f"Score for {Path(f_path).name}: {score:.4f}")    

        print(f"Successfully saved {successful_outputs}/{n_outputs} outputs")
        stats["successful_generations"] += successful_outputs
        stats["failed_generations"] += (n_outputs - successful_outputs)
        stats["output_files"].extend(prompt_files)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI files from text prompts using MIDI-LLM with HuggingFace",    # What it does
        formatter_class=argparse.RawDescriptionHelpFormatter,                                   # Displays the text without auto-formatting it
        epilog="""                                                                              
Examples:
  # Generate from a single prompt (there will be 4 outputs by default)
  python generate_transformers.py --model path/to/checkpoint \\
      --prompt "A cheerful piano melody"
  
  # Generate single output without synthesis
  python generate_transformers.py --model path/to/checkpoint \\
      --prompt "A relaxing jazz piece" \\
      --n_outputs 1 \\
      --no-synthesize
  
  # Interactive mode (with initial prompt)
  python generate_transformers.py --model path/to/checkpoint \\
      --prompt "A cheerful melody" \\
      --interactive
  
  # Interactive-only mode (no initial prompt)
  python generate_transformers.py --model path/to/checkpoint \\
      --interactive
  
  # Generate from prompts file
  python generate_transformers.py --model path/to/checkpoint \\
      --prompts_file prompts.txt
        """                                                                                     # Practical examples
    )

    # Required arguments
    parser.add_argument(
        "--model",  # User label
        type=str,   # Whatever the user types will be treated as a string
        default="slseanwu/MIDI-LLM_Llama-3.2-1B",   # Default model
        help="Path to MIDI-LLM model checkpoint, can be HuggingFace model ID or local path (default: slseanwu/MIDI-LLM_Llama-3.2-1B)"
    )
    
    parser.add_argument(
        "--lora",  # User label
        type=str,   # Whatever the user types will be treated as a string
        default="./lora_groove_midi_model22",  
        help="Path to LoRA adapter"
    )

    """
    Creates mutual exclusion between command line arguments and a text file. 
    required=False allows the script to start even if the user wants to enter prompts manually later (interactive mode).
    """
    
    # Input arguments (not required if using --interactive only) 
    input_group = parser.add_mutually_exclusive_group(required=False) # Sets mutual exclusion
    
    # To generate music from a single phrase typed on the spot
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single text prompt for generation"
    )
    # To read a list of phrases from an external file
    input_group.add_argument(
        "--prompts_file",
        type=str,
        help="Path to file containing prompts (one per line)"
    )

    # Output arguments
    
    # Creates a folder named generated_outputs in the same location as the script
    parser.add_argument(
        "--output_root",
        type=str,
        default="./generated_outputs",
        help="Root directory for outputs (timestamped subdirs will be created inside, default: ./generated_outputs)"
    )
    # Indicates the number of output files generated from the same prompt
    parser.add_argument(
        "--n_outputs",
        type=int,
        default=DEFAULT_N_OUTPUTS,
        help=f"Number of outputs to generate per prompt (default: {DEFAULT_N_OUTPUTS})"
    )

    # Synthesis arguments
    
    # Changes the synthesize variable to false
    parser.add_argument(
        "--no-synthesize",
        dest="synthesize",      # Decides the variable name, which by default would be args.no_synthesize
        action="store_false",   # Decides the value
        help="Skip audio synthesis (only generate MIDI files)"
    )
    
    parser.set_defaults(synthesize=True)    # Sets the default to true
    
    # Adds the soundfont, the default one is FluidR3
    parser.add_argument(
        "--soundfont",
        type=str,
        default="./soundfonts/FluidR3_GM/FluidR3_GM.sf2",
        help="Path to SoundFont file for synthesis (default: ./soundfonts/FluidR3_GM/FluidR3_GM.sf2)"
    )

    # Generation parameters, set the default values
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=DEFAULT_TOP_P,
        help=f"Nucleus sampling threshold (default: {DEFAULT_TOP_P})"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )

    # Model arguments
    
    # Tells HuggingFace to download the model into a specific folder
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: $HF_HOME or ~/.cache/huggingface)"
    )
    
    # If enabled, the script doesn't close after the first generation but stays active until an empty prompt is received
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after initial generation (keep generating until empty prompt)"
    )

    parser.add_argument(
    "--use_clap",
    action="store_true",
    help="Use CLAP model to evaluate generated audio similarity"
    )
    
   # Transforms a long, unorganized text string from the terminal into a clean, organized object for the script to use
    args = parser.parse_args()
    
    

    # Validate that either prompts are provided or interactive mode is enabled
    if not args.prompt and not args.prompts_file and not args.interactive:
        parser.error("Either --prompt, --prompts_file, or --interactive must be specified")

    # Load prompts (if provided)
    prompts = []    # Initializes an empty list
    if args.prompt:     # If the prompt is entered via command line
        prompts = [args.prompt]     # It is added to the list
    elif args.prompts_file:         # If the prompt is contained in a text file
        with open(args.prompts_file, "r") as f: # The file is opened in read mode
            prompts = [line.strip() for line in f if line.strip()]  # For each line, removes useless spaces and \n at the beginning and end
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}") # Prints that the prompt was loaded

    # Check synthesis requirements
    if args.synthesize:     # If the audio needs to be synthesized
        soundfont_path = Path(args.soundfont)   # Transforms the text into a path
        if not soundfont_path.exists():     # If the soundfont was not found
            print(f"Error: SoundFont not found at {soundfont_path}")    # Prints the message
            print("Please download a SoundFont or disable synthesis")  
            import sys
            sys.exit(1) # Closes the program

    # Create output root directory with timestamp
    output_root = Path(args.output_root)    # Transforms the text into an output path
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # Saves the current timestamp
    output_dir = output_root / session_timestamp    # Creates the directory name
    output_dir.mkdir(parents=True, exist_ok=True)   # Creates the directory
    # If the main folder doesn't exist, the program creates it automatically; if it already exists, the program won't crash

    print(f"Output directory: {output_dir.absolute()}\n")   # Prints the absolute path of the created folder

    # Load tokenizer from the model checkpoint
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,                 # Loads the specific configuration of the inputted model
        cache_dir=args.cache_dir,   # Where to save the model files once downloaded
        pad_token="<|eot_id|>",     # End of text token
    )

    # Load model
    model = prepare_hf_model(model_path=args.model, lora_path=args.lora)     # Loads the model using the previously defined function
    
    if args.synthesize and args.use_clap:
        init_clap()

    # Generate from initial prompts (if provided)
    if prompts: # Checks if the prompts list contains anything
        print(f"Starting generation for {len(prompts)} prompt(s)...\n") 
        start_time = time.time()   # Starts the stopwatch 

        # The function's result is saved in the stats variable
        stats = generate_from_prompts_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_dir,
            model_path=args.model,
            soundfont_path=args.soundfont if args.synthesize else None, # If synthesis is disabled, sends None instead of the file path, saving memory
            synthesize=args.synthesize,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n_outputs=args.n_outputs,
            use_clap=args.use_clap
        )

        total_time = time.time() - start_time   # Calculates the total time spent generating

        # Print summary
        print(f"\n{'='*70}")
        print("Generation Summary")
        print(f"{'='*70}")
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Successful generations: {stats['successful_generations']}")
        print(f"Failed generations: {stats['failed_generations']}")
        print(f"Total time: {total_time:.2f}s")

        if stats['generation_times']:   # Verifies that the times list is not empty
            avg_time = sum(stats['generation_times']) / len(stats['generation_times'])  # Calculates the average time
            print(f"Average generation time: {avg_time:.2f}s (excluding warmup)")       # Prints the average time

        print(f"\nOutputs saved to: {output_dir.absolute()}")

        # Print generated files
        if stats['output_files']:       # Checks if any output files were generated
            print(f"\nGenerated files:")   # Prints the generated files 
            for file_path in stats['output_files']:
                file_type = "< MIDI" if file_path.endswith('.mid') else "< Audio"
                print(f"  {file_type}: {file_path}")

        print(f"{'='*70}\n")

        # Save stats to JSON
        stats_file = output_dir / "generation_stats.json"   # Saves the statistics in a json file
        with open(stats_file, "w") as f:
            json.dump({
                **stats,
                "total_time": total_time,
                "average_time": sum(stats['generation_times']) / len(stats['generation_times']) if stats['generation_times'] else 0,
                "config": {
                    "model": args.model,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "n_outputs": args.n_outputs,
                }
            }, f, indent=2)
    else:   # If the user didn't enter a prompt
        print(f"No initial prompts provided. Starting in interactive mode...\n")

    # Interactive mode
    if args.interactive:
        print(f"\n{'='*70}")
        print("Interactive Mode")
        print(f"{'='*70}")
        print("Enter prompts to generate more MIDI files.")
        print("Press Enter with empty prompt to exit.\n")

        while True:
            try:
                # Get user input
                user_prompt = input("Prompt: ").strip() # Takes keyboard input, removing accidental \n at the beginning or end of the text

                # Exit if empty
                if not user_prompt: # If the prompt is empty
                    print("\nExiting interactive mode. Goodbye!")
                    break

                # Generate from the new prompt
                print()
                # Every time you press Enter and the AI generates music, the function returns a stats dictionary specific to that single interaction.
                interactive_stats = generate_from_prompts_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[user_prompt], # The program takes the single phrase just typed (user_prompt) and puts it in a list.
                    output_dir=output_dir,
                    model_path=args.model,
                    soundfont_path=args.soundfont if args.synthesize else None,
                    synthesize=args.synthesize,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    n_outputs=args.n_outputs
                )

                # Print input prompt
                print(f"Input prompt: {user_prompt}")

                # Print mini summary
                print(f"\n Generated {interactive_stats['successful_generations']}/{args.n_outputs} outputs")
                if interactive_stats['generation_times']:
                    print(f"  Generation time: {interactive_stats['generation_times'][0]:.2f}s")

                # Print file paths
                if interactive_stats['output_files']:
                    for file_path in interactive_stats['output_files']:
                        file_type = "<" if file_path.endswith('.mid') else "<"
                        print(f"  {file_type} {file_path}")
                print()

            except KeyboardInterrupt: # Catches the signal sent when the Ctrl + C key combination is pressed on the keyboard.
                print("\n\nInterrupted. Exiting interactive mode.")
                break
            except EOFError: # Catches an end of file (EOF)
                print("\n\nExiting interactive mode.")
                break

# Checks how the file was opened; if imported into another project, __name__ takes the file's name and not __main__
if __name__ == "__main__":
    main()