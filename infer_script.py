from datetime import datetime

import torch
import pandas as pd
import time
import logging
import sys
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# ---------------------------------------------------------
# 1. Logging Setup
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE = f"./logs/inference_log_{timestamp}.txt"

# Set up logging to output to both console and log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 2. Configuration
# ---------------------------------------------------------
BASE_MODEL_NAME = "LLM4Binary/llm4decompile-9b-v2" 
ADAPTER_DIR = "./final_lora_model"
DATASET_PATH = "./final_training_dataset"
OUTPUT_SAVE_PATH = f"./inference_results_dataset_{timestamp}"
# IMPORTANT: Change this to the column that actually holds your input text!
INPUT_COLUMN_NAME = "instruction" 

def main():
    total_start_time = time.time()
    logger.info("=== STARTING INFERENCE SCRIPT ===")
    
    # ---------------------------------------------------------
    # 3. Model & Tokenizer Initialization
    # ---------------------------------------------------------
    logger.info("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info(f"Loading tokenizer and adapter from {ADAPTER_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval() # Set model to evaluation mode

    # ---------------------------------------------------------
    # 4. Load Dataset
    # ---------------------------------------------------------
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    ogdataset = dataset.to_pandas()
    ogdataset = ogdataset[ogdataset["status"].eq("Success")].copy()
    total_rows = len(ogdataset)
    logger.info(f"Dataset loaded successfully. Total rows to process: {total_rows}")

    # ---------------------------------------------------------
    # 5. Inference Loop with Partial Save Handling
    # ---------------------------------------------------------
    results = []
    
    logger.info("Starting inference loop. Press Ctrl+C at any time to safely stop and save progress.")
    
    # Create the progress bar
    pbar = tqdm(total=total_rows, desc="Processing Prompts", unit="prompt")
    
    try:
        for index, row in ogdataset.iterrows():
            # Track time for this specific prompt
            prompt_start_time = time.time()
            
            user_input = row[INPUT_COLUMN_NAME]
            extracted_code= row['extracted_code']
            prompt = f"""<|im_start|>system
You are an expert programming assistant. You must always think step-by-step inside <scratchpad> tags before providing your final answer.<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
            
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Slice the output to only get the new generated tokens
            input_length = inputs.input_ids.shape[-1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
            
            prompt_end_time = time.time()
            prompt_duration = prompt_end_time - prompt_start_time
            
            # Store the result including execution time
            results.append({
                'original_row_index': index,
                'prompt': prompt,
                'output': response,
                'time_taken_seconds': round(prompt_duration, 4),
                'extracted_code': extracted_code
            })
            
            # Update progress bar
            pbar.update(1)
            
    except KeyboardInterrupt:
        # This block catches the Ctrl+C cancellation
        logger.warning("\nKeyboardInterrupt detected! Halting inference early...")
        logger.warning(f"Successfully processed {len(results)} out of {total_rows} records before interruption.")
        
    except Exception as e:
        # Catches out-of-memory (OOM) or other unexpected errors, saving what you have so far
        logger.error(f"\nAn error occurred during inference: {str(e)}")
        logger.warning(f"Attempting to save the {len(results)} records completed so far...")
        
    finally:
        # This block ALWAYS runs, whether it finishes, is cancelled, or crashes
        pbar.close()
        
        if len(results) > 0:
            logger.info("Converting results to HuggingFace dataset...")
            results_df = pd.DataFrame(results)
            final_dataset = Dataset.from_pandas(results_df)

            logger.info(f"Saving dataset to {OUTPUT_SAVE_PATH}...")
            final_dataset.save_to_disk(OUTPUT_SAVE_PATH)
            logger.info("Dataset successfully saved!")
        else:
            logger.warning("No results were generated. Nothing to save.")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        
        # Calculate human-readable time (hours, minutes, seconds)
        hours, rem = divmod(total_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"
        
        logger.info(f"=== SCRIPT COMPLETED ===")
        logger.info(f"Total rows processed: {len(results)}")
        logger.info(f"Total time taken (Start to Finish): {time_str}")

if __name__ == "__main__":
    main()