import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# HYPERPARAMETERS & CONFIGURATION
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-Coder-32B-Instruct"
DATASET_PATH = "./final_training_dataset"   
OUTPUT_DIR = "./results"
FINAL_MODEL_DIR = "./final_lora_model"

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 4096

def main():
    print("Loading dataset...")
    # Load your already-formatted dataset
    dataset = load_from_disk(DATASET_PATH)

    # Split train/val
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
        device_map="auto",
        max_memory={0: "14GiB", 1: "22GiB", 2: "22GiB", 3: "22GiB"},
        use_safetensors=True,
        attn_implementation="sdpa"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # ==========================================
    # LORA CONFIGURATION
    # ==========================================
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # ==========================================
    # TRAINING CONFIGURATION
    # ==========================================
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        num_train_epochs=NUM_EPOCHS,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        optim="paged_adamw_8bit",
        bf16=True,
        run_name="qwen-bytecode-cot",
        report_to="none",
        loss_type="nll"
    )

    # ==========================================
    # INITIALIZE & START TRAINING
    # ==========================================
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training!")
    trainer.train()

    print(f"Saving final model adapter to {FINAL_MODEL_DIR}...")
    trainer.model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print("Done!")

if __name__ == "__main__":
    main()