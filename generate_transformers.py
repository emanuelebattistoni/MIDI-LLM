#!/usr/bin/env python3
"""
MIDI-LLM: Text-to-MIDI Generation using HuggingFace Transformers

This script generates MIDI files from text prompts using the MIDI-LLM model with HuggingFace backend.
Simpler to set up than vLLM but slower for inference.
"""

import json # Serve per leggere e scrivere dati in formato JSON
import time # Gestisce il tempo
import argparse # Serve per creare interfacce a riga di comando
from pathlib import Path # Modo moderno e orientato agli oggetti per gestire i percorsi dei file e delle cartelle
from datetime import datetime # Serve per ottenere la data e l'ora correnti
from typing import List, Optional # Aiutano chi legge il codice a capire che tipo di dato si aspetta una funzione, 
#  optional indica che un parametro pu� anche essere nullo (None)

import torch # Framework principale usato per il calcolo tensoriale e il deep learning
import tqdm # Libreria puramente visiva
from transformers import AutoTokenizer, AutoModelForCausalLM # Classi principali della libreria di Hugging Face
from transformers import StoppingCriteria, StoppingCriteriaList    #aggiunto per la barra di caricamento 
# AutoTokenizer: Traduce il testo umano in una sequenza di(token)
# AutoModelForCausalLM: Carica i pesi del modello vero e proprio

# Import helper functions and constant
from midi_llm.utils import (
    save_generation,    # Save generated tokens as MIDI file
    AMT_GPT2_BOS_ID,    #55026
    LLAMA_VOCAB_SIZE,   #128256
    LLAMA_MODEL_NAME,   #meta-llama/Llama-3.2-1B
)

import mido

# Mappa standard General MIDI
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

def get_instruments_from_tokens(tokens: List[int]) -> List[str]:
    """Estrae i nomi degli strumenti analizzando i token in memoria."""
    found_instruments = set()
    for i in range(len(tokens) - 1):
        # I byte 192-207 indicano un cambio strumento sui 16 canali MIDI
        if 192 <= tokens[i] <= 207:
            instr_id = tokens[i + 1]
            if 0 <= instr_id <= 127:
                name = GM_INSTRUMENTS.get(instr_id, f"Unknown ({instr_id})")
                found_instruments.add(name)
    
    # Ritorna Grand Piano se la lista è vuota (default MIDI)
    return list(found_instruments) if found_instruments else [GM_INSTRUMENTS[0]]

class ProgressMonitor(StoppingCriteria):
    def __init__(self, max_new_tokens, prompt_length):
        self.prompt_length = prompt_length
        # Creiamo la barra con un formato puramente grafico e di velocità
        self.pbar = tqdm.tqdm(
            total=max_new_tokens, 
            desc="  ↳ Generating:", 
            position=1, 
            leave=False,
            # {bar} crea i blocchi, rimuoviamo {n_fmt} (i numeri nudi)
            bar_format="{desc}: {percentage:3.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Calcola quante note sono state scritte
        current_tokens = input_ids.shape[1] - self.prompt_length
        increment = current_tokens - self.pbar.n
        
        # Fa avanzare la barra visiva
        if increment > 0:
            self.pbar.update(increment)
            
        return False 
        
    def close(self):
        self.pbar.close()

# Default generation parameters
DEFAULT_TEMPERATURE = 1.0   #Le probabilita' si "appiattiscono". Il modello inizia a dare una chance anche a note meno scontate
DEFAULT_TOP_P = 0.98    #Nucleus Sampling, scarta a priori il 2% delle note piu' assurde e improbabili
DEFAULT_MAX_TOKENS = 2046   #Lunghezza massima della composizione generata
DEFAULT_N_OUTPUTS = 4  # give more outputs for variability

#Caricamento del modello
def prepare_hf_model(model_path: str):
    """
    Initialize HuggingFace model in BFloat16.
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        model
    """
    #Viene stampato un log visivo
    print(f"\n{'='*70}")
    print("Model Configuration")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Precision: BFloat16")
    print(f"{'='*70}\n")
    
    # Load model in BF16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,             # Path to model checkpoint
        dtype=torch.bfloat16,   # DataType dei parametri del modello
        trust_remote_code=True  # Diamo il permesso al computer di eseguire il codice che ha allegato al modello
    ).to(device="cuda")         # Trasferisce l'intero modello nella VRAM della scheda video NVIDIA
    
    model.eval()    # Passa alla modalita' di inferenza congelando i pesi
    print(f" Model loaded successfully\n")
    
    return model    # Restituisce il modello intero, gia' caricato nella GPU e pronto all'uso


def generate_from_prompts_hf(
   #prende in input 
    model,                      # Il modello caricato
    tokenizer: AutoTokenizer,   # Il tokenizer del modello caricato
    prompts: List[str],         # Lista dei prompt scritti
    output_dir: Path,           # Path della cartella principale dove la funzione dovra' creare le sottocartelle per salvare i file
    model_path: str,            # Path per il modello da caricare
    soundfont_path: Optional[str] = None,
    synthesize: bool = False,   # Dato che e' false la funzione non tentera' di convertire il file MIDI in mp3.
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    n_outputs: int = DEFAULT_N_OUTPUTS,
    system_prompt: Optional[str] = None # Dato che e' impostato a None, il codice utilizzera' quello di default
) -> dict:
    """
    Generate MIDI from text prompts using HuggingFace model.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of text prompts
        output_dir: Base output directory
        model_path: Path to model (to check for with_edits)
        soundfont_path: Path to SoundFont file
        synthesize: Whether to synthesize to audio
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_tokens: Maximum tokens to generate
        n_outputs: Number of outputs per prompt
        system_prompt: Optional system prompt prefix
        
    Returns:
        Dictionary with generation statistics and output files
    """
    # Default system prompt--> da modificare quando viene rifatto il training
    if system_prompt is None:
        system_prompt = "You are a world-class composer. Please compose some music according to the following description: "
    
    stats = {
        "total_prompts": len(prompts),  # Conta il numero di prompt inseriti
        "successful_generations": 0,    # Inizializza le generazioni avvenute con successo a 0
        "failed_generations": 0,        # Inizializza le generazioni avvenute senza successo a 0
        "generation_times": [],         # Crea una lista vuota dove salvera' il tempio impiegato per le generazioni
        "output_files": [],              # Crea una lista vuota in cui andra' a salvare i percorsi esatti
        "midi_instruments": {}          #aggiunge una lista vuota per gli strumenti utilizzati    
    }
    
    """Avvia il ciclo:
    - tqdm(prompts): Genera una barra di caricamento prendendo la length della lista di prompt ed ogni volta che ne viene processato uno
    aggiorna la barra di caricamento facendola aumentare
    - enumerate: associa un indice ad ogni prompt della lista partendo da zero """
    
    for idx, prompt in enumerate(tqdm.tqdm(prompts, desc="Generating")):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")   # Stampa quale prompt sta elaborando dei totali, compreso del contenuto
        
        # Create output directory for this prompt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # Salva la data e l'orario di ricezione del prompt
        prompt_output_dir = output_dir / f"{timestamp}_prompt_{idx+1}" # Crea il path della directory dove verr� salvato l'output del prompt
        
        # Prepare full prompt, add space to the end of each prompt to match training
        full_prompt = system_prompt + prompt + " "
            
        # Tokenize
        
        """Tokenizer prende in input il prompt, lo tokenizza secondo il tokenizzatore del modello e ritorna un tensore di PyTorch
        senza eseguire lo zero padding, input_ids invece va a selezionare la sequenza numerica vera e propria tralasciando
        l'attention mask"""
        
        llama_input = tokenizer(full_prompt, return_tensors="pt", padding=False)
        input_ids = llama_input["input_ids"]
        
        # Add MIDI BOS token
        """ Viene calcolato il token MIDI Beginning of Sequence e viene trasformato in tensore dalla torch.tensor, in modo che venga
        concatenato poi dalla torch.cat a quello che e' il tensore ricavato precedentemente dal tokenizer. Cosi' il modello e' obbligato
        a generare un nota musicale poiche' grazie alla fase di training e' addestrato a procedere cosi'. Dim e' impostata a 1 per far si che
        il midi_bos venga concatenato in colonna data la struttura del dato (Batch, Sequenza), allungando quindi la sequenza e non creando
        un ulteriore riga"""
        
        midi_bos = torch.tensor([[AMT_GPT2_BOS_ID + LLAMA_VOCAB_SIZE]]) # Due [] per le dimensioni
        input_ids = torch.cat([input_ids, midi_bos], dim=1) #concateno in colonna
        
        # Move to device
        """ Il comando model.parameters() restituisce un generatore(un oggetto che contiene tutti i parametri),
        next serve ad estrarre il primo elemento disponibile da un generatore,
        .device serve a capire se quel componente e' su CPU o GPU e tale informazione viene salvata in "device
        .to(device) serve a spostare i dati dove e' stato caricato il modello """
      
        
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)    # Spostiamo i dati dove e' stato caricato il modello
        
        # Generate multiple outputs
        start_time = time.time()    # Start del cronometro per il benchmarking
        
        # 1. Misura quanti token occupa la nostra richiesta testuale
        prompt_len = input_ids.shape[1]
        
        # 2. Passa ENTRAMBI i valori al monitor (limite massimo e lunghezza iniziale)
        monitor = ProgressMonitor(max_tokens, prompt_len) # Inizializza il monitor passando i token massimi

        with torch.no_grad():       # Passa alla modalita' inferenza, evitando di salvare i passaggi intermedi
            outputs = model.generate(
                input_ids=input_ids,    # Passa al modello il tensore costruito fino a questo punto
                do_sample=True,         # Se settato su false resttuisce solo la nota pi� probabile, cos� varia
                max_new_tokens=max_tokens,  
                temperature=temperature,    
                top_p=top_p,
                num_return_sequences=n_outputs,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([monitor])  #aggiunta per la barra di avanzamento 
            )

        monitor.close() # Chiude la barra secondaria quando ha finito
        
        generation_time = time.time() - start_time  # Calcola il tempo impiegato nella generazione
        
        if idx > 0:  # Skip first generation for timing (warmup),  per evitare che il tempo loading del modello venga misurato
            stats["generation_times"].append(generation_time) # Aggiunge alla lista dei tempi, il tempo di generazione per ogni prompt
        
        print(f"Generation time: {generation_time:.2f}s") # Va a stampare il tempo di generazione
        
        # Extract only the generated tokens (remove prompt)
        prompt_len = input_ids.shape[1]     # Misura la lunghezza(in token) della richiesta iniziale
        outputs = outputs[:, prompt_len:]   # Prende in considerazione solamente i token generati dall'ai
        
        # Shift tokens back to MIDI vocab range
        outputs = outputs - LLAMA_VOCAB_SIZE    # L'output viene shiftato nel range MIDI per far si che venga riconosciuta la nota
        outputs = outputs.cpu().tolist()        # I dati vengono riportati nella cpu e convertiti in una lista
        
        # Save all outputs for this prompt
        successful_outputs = 0      #inizializza il contatore degli output con successo a 0
        prompt_files = []          # Inizializza una lista per salvare dopo in essa il path di tutto ci� che stiamo per salvare riguardo allo specifico prompt
        
        
        #outputs e' la lista di tutte le melodie che ha generato l'ia,  
        
        for output_idx, midi_tokens in enumerate(outputs):
            # Save generation, Save generated tokens as MIDI file
            success = save_generation(
               #input 
                tokens=midi_tokens,     #Lista di numeri (note) puliti costruiti in precedenza
                prompt=prompt,          #descrizione testuale originale
                output_dir=prompt_output_dir,   #Cartella specifica nella quale deve essere salvato il prompt
                generation_idx=output_idx + 1,  #Serve a numerare il file
                soundfont_path=soundfont_path,  #Percorso al file .s2 per lo "strumento musicale virtuale"
                synthesize=synthesize           #Se sintetizzare il file o meno
            )
            
            if success:
                successful_outputs += 1
                midi_file = prompt_output_dir / f"gen_{output_idx + 1}.mid"
                prompt_files.append(str(midi_file))
                
                try:
                    # Analisi diretta dei token per trovare i Program Change
                    nomi_strumenti = get_instruments_from_tokens(midi_tokens)
                    
                    # Usiamo il percorso relativo per evitare sovrascritture nel JSON
                    relative_path = str(midi_file.relative_to(output_dir))
                    stats["midi_instruments"][relative_path] = nomi_strumenti
                except Exception as e:
                    stats["midi_instruments"][str(midi_file.name)] = f"Errore: {str(e)}"
                    
                if synthesize and soundfont_path:                           #se attiva l'opzione viene costruito il nome del file .mp3
                    mp3_file = prompt_output_dir / f"gen_{output_idx + 1}.mp3"
                    if mp3_file.exists():
                        prompt_files.append(str(mp3_file))                  #viene aggiunto alla lista dei file
        
        print(f"Successfully saved {successful_outputs}/{n_outputs} outputs")
        stats["successful_generations"] += successful_outputs
        stats["failed_generations"] += (n_outputs - successful_outputs)
        stats["output_files"].extend(prompt_files)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI files from text prompts using MIDI-LLM with HuggingFace",    # Che cosa fa
        formatter_class=argparse.RawDescriptionHelpFormatter,                                   # Visualizza il testo senza formattarlo
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
        """                                                                                     # Esempi pratici
    )

    # Required arguments
    parser.add_argument(
        "--model",  #Etichetta per l'utente
        type=str,   #qualsiasi cosa l'utente scriva verr� trattata come stringa
        default="slseanwu/MIDI-LLM_Llama-3.2-1B",   #modello di default
        help="Path to MIDI-LLM model checkpoint, can be HuggingFace model ID or local path (default: slseanwu/MIDI-LLM_Llama-3.2-1B)"
    )

    """Crea mutua esclusione tra i comandi dalla riga di comando e i comandi passati da un file di testo, con required = false permette
    all'utente di non scegliere la modalita' permettendo allo script di avviarsi ache se l'utente vuole inserire i prompt a mano in un
    secondo momento"""
    
    # Input arguments (not required if using --interactive only) 
    input_group = parser.add_mutually_exclusive_group(required=False) # Imposta la mutua esclusione
    
    # Per generare musica da una singola frase scritta al momento
    input_group.add_argument(
        "--prompt",
        type=str,
        help="Single text prompt for generation"
    )
    #Per leggere una lista di frasi da un file esterno
    input_group.add_argument(
        "--prompts_file",
        type=str,
        help="Path to file containing prompts (one per line)"
    )

    # Output arguments
    
    # Crea una cartella chiamata generated_outputs nella stessa posizione dello script
    parser.add_argument(
        "--output_root",
        type=str,
        default="./generated_outputs",
        help="Root directory for outputs (timestamped subdirs will be created inside, default: ./generated_outputs)"
    )
    # Indica il numero di file di output che verranno generati dallo stesso prompt
    parser.add_argument(
        "--n_outputs",
        type=int,
        default=DEFAULT_N_OUTPUTS,
        help=f"Number of outputs to generate per prompt (default: {DEFAULT_N_OUTPUTS})"
    )

    # Synthesis arguments
    
    #Cambia la variabile synthetize in false
    parser.add_argument(
        "--no-synthesize",
        dest="synthesize",      #Decide il nome della variabile, che di default in questo caso sarebbe args.no_synthesize
        action="store_false",   #Decide il valore
        help="Skip audio synthesis (only generate MIDI files)"
    )
    
    parser.set_defaults(synthesize=True)    #setta il default a true
    
    #aggiunge il soundfont, quello di base e' FluidR3
    parser.add_argument(
        "--soundfont",
        type=str,
        default="./soundfonts/FluidR3_GM/FluidR3_GM.sf2",
        help="Path to SoundFont file for synthesis (default: ./soundfonts/FluidR3_GM/FluidR3_GM.sf2)"
    )

    # Generation parameters, setta i valori di default
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
    
    # Indica ad HuggingFace di scaricare il modello in una cartella specifica
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: $HF_HOME or ~/.cache/huggingface)"
    )
    
    #Se viene attivata lo script non si chiude dopo la prima generazione ma rimane attivo fino a che non ricevo un prompt vuoto
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive mode after initial generation (keep generating until empty prompt)"
    )
   # trasforma una lunga stringa di testo scritta disordinatamente nel terminale in un oggetto pulito e organizzato che lo script puo' usare per lavorare
    args = parser.parse_args()
    
    

    # Validate that either prompts are provided or interactive mode is enabled
    if not args.prompt and not args.prompts_file and not args.interactive:
        parser.error("Either --prompt, --prompts_file, or --interactive must be specified")

    # Load prompts (if provided)
    prompts = []    # Inizializza una lista vuota
    if args.prompt:     # Se il prompt viene inserito da riga di comando
        prompts = [args.prompt]     # Viene aggiunto alla lista
    elif args.prompts_file:         # Se il prompt � contenuto in un file testuale
        with open(args.prompts_file, "r") as f: # Viene aperto il file in lettura
            prompts = [line.strip() for line in f if line.strip()]  # Per ogni riga toglie gli spazi inutili e gli \n all'inizio e alla fine
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}") # Stampa che il prompt viene aggiunto alla lista

    # Check synthesis requirements
    if args.synthesize:     # Se l'audio va sintetizzato in audio
        soundfont_path = Path(args.soundfont)   # Trasfroma il testo in un path
        if not soundfont_path.exists():     #se il sountfont non � stato inserito
            print(f"Error: SoundFont not found at {soundfont_path}")    #stampa il messaggio
            print("Please download a SoundFont or disable synthesis")  
            import sys
            sys.exit(1) # Chiude il programma

    # Create output root directory with timestamp
    output_root = Path(args.output_root)    # Trasforma il testo in un path di output
    session_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")  # Salva il time stamp corrente
    output_dir = output_root / session_timestamp    # Crea il nome della directory
    output_dir.mkdir(parents=True, exist_ok=True)   # Crea la directory
    # Se la cartella principale non esiste, il programma la crea automaticamente, se la cartella esiste gi� il programma non crasha

    print(f"Output directory: {output_dir.absolute()}\n")   # Stampa il percorso assoluto della cartella creata

    # Load tokenizer from the model checkpoint
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,                 # Carica la configurazione specifica del modello inserito
        cache_dir=args.cache_dir,   # Dove salvare i file del modello una volta scaricati.
        pad_token="<|eot_id|>",     # Token di fine testo
    )

    # Load model
    model = prepare_hf_model(model_path=args.model)     # Carica il , modello utilizzando la funzione definita in precedenza

    # Generate from initial prompts (if provided)
    if prompts: # Controlla se la lista dei prompts contiene qualcosa
        print(f"Starting generation for {len(prompts)} prompt(s)...\n") 
        start_time = time.time()   #fa partire il cronometro 

        # Il risultato della funzione viene salvato nella variabile stats
        stats = generate_from_prompts_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            output_dir=output_dir,
            model_path=args.model,
            soundfont_path=args.soundfont if args.synthesize else None, # Se la sintesi e' disattivata, invia None invece del percorso del file, risparmiando memoria
            synthesize=args.synthesize,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n_outputs=args.n_outputs
        )

        total_time = time.time() - start_time   # Calcola il tempo impiegato nella generazione

        # Print summary
        print(f"\n{'='*70}")
        print("Generation Summary")
        print(f"{'='*70}")
        print(f"Total prompts: {stats['total_prompts']}")
        print(f"Successful generations: {stats['successful_generations']}")
        print(f"Failed generations: {stats['failed_generations']}")
        print(f"Total time: {total_time:.2f}s")

        if stats['generation_times']:   #verifica che la lista dei tempi non sia vuota
            avg_time = sum(stats['generation_times']) / len(stats['generation_times'])  #calcola la media dei tempi
            print(f"Average generation time: {avg_time:.2f}s (excluding warmup)")       #stampa la media dei tempo

        print(f"\nOutputs saved to: {output_dir.absolute()}")

        # Print generated files
        if stats['output_files']:       #controlla se ci sono dei file di output generati
            print(f"\nGenerated files:")   #stampa i file generati 
            for file_path in stats['output_files']:
                file_type = "< MIDI" if file_path.endswith('.mid') else "< Audio"
                print(f"  {file_type}: {file_path}")

        print(f"{'='*70}\n")

        # Save stats to JSON
        stats_file = output_dir / "generation_stats.json"   #salva le statistiche in un file json
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
    else:   #se l'utente non ha inserito un prompt
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
                user_prompt = input("Prompt: ").strip() #prende input da tastiera andando a eliminare gli \n accidentali all'inizio o alla fine del testo

                # Exit if empty
                if not user_prompt: # Se il prompt e' vuoto
                    print("\nExiting interactive mode. Goodbye!")
                    break

                # Generate from the new prompt
                print()
                #Ogni volta che premi Invio e l'IA genera musica, la funzione restituisce un dizionario di statistiche specifico per quella singola interazione.
                interactive_stats = generate_from_prompts_hf(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=[user_prompt],#l programma prende l'unica frase appena digitata (user_prompt) e la mette in una lista.
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
                        file_type = "<✓" if file_path.endswith('.mid') else "<✓"
                        print(f"  {file_type} {file_path}")
                print()

            except KeyboardInterrupt: #intercetta il segnale inviato quando viene premuta la combinazione di tasti Ctrl + C sulla tastiera.
                print("\n\nInterrupted. Exiting interactive mode.")
                break
            except EOFError:#intercetta un end of file
                print("\n\nExiting interactive mode.")
                break

#Verifica come il file � stato aperto, se viene importato il file in un altro progetto, __name__ assumere il nome del file e non __main__
if __name__ == "__main__":
    main()