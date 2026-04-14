import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling  
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME   = "slseanwu/MIDI-LLM_Llama-3.2-1B"
DATASET_PATH = "./data/groove_sft_dataset_hf.jsonl"
OUTPUT_DIR   = "./lora_groove_midi_model"
MAX_LENGTH   = 2048  

def main():
    print("1. Caricamento del Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|eot_id|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    print("2. Caricamento del Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    if "labels" in dataset.column_names:
        dataset = dataset.remove_columns(["labels"])
        
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def filter_long(example):
        return len(example["input_ids"]) <= MAX_LENGTH

    train_dataset = split_dataset["train"].filter(filter_long)
    eval_dataset  = split_dataset["test"].filter(filter_long)
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    print("3. Caricamento del Modello in BF16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to("cuda") 

    print("4. Configurazione di LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads() 
    model.print_trainable_parameters()

    print("5. Configurazione del Training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,   
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3.05,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,     
        optim="adamw_torch_fused",       
        report_to="tensorboard"
    )

    data_collator = DataCollatorForLanguageModeling( 
        tokenizer=tokenizer,
        mlm=False,
    )

    print("6. Avvio dell'addestramento...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train()

    print("7. Salvataggio del modello LoRA...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Addestramento completato! Adattatori LoRA salvati in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()