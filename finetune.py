import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"  # Extremely lightweight, perfect for laptops
DATASET_PATH = "dataset.jsonl"
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./final_lora_model"

# You can adjust these for lower/higher VRAM usage
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048

def format_instruction(sample):
    """
    Format the prompt using an Alpaca-style template for instruction tuning.
    """
    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    return prompt

def main():
    print("Loading dataset...")
    dataset = load_dataset('json', data_files={'train': DATASET_PATH}, split='train')
    
    # Optionally split train/val
    dataset = dataset.train_test_split(test_size=0.05)
    train_data = dataset['train']
    val_data = dataset['test']
    print(f"Train size: {len(train_data)} | Val size: {len(val_data)}")

    # ==========================================
    # LOAD MODEL WITH QLORA (4-bit)
    # ==========================================
    print("Loading model and tokenizer...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # ==========================================
    # LORA CONFIGURATION
    # ==========================================
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)

    # ==========================================
    # TRAINING
    # ==========================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        max_steps=-1,
        num_train_epochs=NUM_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit",
        fp16=True, 
        run_name="pyc-finetune",
        report_to="none" # change to "wandb" if you use weights and biases
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=lambda example: [format_instruction(ex) for ex in zip(*[example[k] for k in example])], # This deals with batching
    )

    # Actually we can map the dataset directly to text to avoid formatting_func bugs
    def map_formatting(example):
        example['text'] = format_instruction(example)
        return example
        
    train_data = train_data.map(map_formatting)
    val_data = val_data.map(map_formatting)

    # Re-initialize trainer using dataset text field
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving final model adapter to {FINAL_MODEL_DIR}...")
    trainer.model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
