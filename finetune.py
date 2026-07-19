import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

def main():
    # 1. Initialize Logging
    # Wandb gives you a beautiful web dashboard for training stats
    # wandb.init(project="bytecode-llm", name="qwen-32b-cot-run")

    # 2. Load Your Prepared Chain of Thought Dataset
    print("Loading dataset...")
    dataset = load_from_disk('./final_training_dataset')

    # 3. Load Tokenizer & Model
    print("Loading model and tokenizer...")
    model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with bfloat16 and Flash Attention 2 to fit in VRAM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", # Automatically distributes across your GPUs
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    # 4. Configure LoRA (Low-Rank Adaptation)
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=64,                # High rank to capture complex logic mapping
        lora_alpha=128,
        target_modules="all-linear", # Target all linear layers for maximum capability
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() # Shows how many parameters you are actually training

    # 5. Set Training Arguments
    training_args = SFTConfig(
        output_dir="./bytecode_cot_model_checkpoints",
        per_device_train_batch_size=4,   # Reduce to 2 or 1 if you get Out Of Memory (OOM) errors
        gradient_accumulation_steps=8,   # Effective batch size = batch_size * gradient_steps * GPUs
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        num_train_epochs=3,              # 3 full passes over your dataset
        bf16=True,                       # Fast mixed-precision training
        optim="adamw_torch_fused",
        report_to="none",
        save_strategy="epoch",  
        dataset_text_field="text",       # Points to the 'text' column we created earlier
        max_length=4096,             # Large enough to hold your scratchpad + bytecode
        packing=False                # Save a checkpoint at the end of every epoch
    )

    # 6. Initialize SFTTrainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )

    # 7. Start Training
    print("Starting training!")
    trainer.train()

    # 8. Save the final adapter model
    print("Saving final model...")
    trainer.model.save_pretrained("./final_bytecode_lora_cot")
    tokenizer.save_pretrained("./final_bytecode_lora_cot")
    
    # wandb.finish()
    print("Training complete!")

if __name__ == '__main__':
    main()