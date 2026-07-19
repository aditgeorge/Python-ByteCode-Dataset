import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-base"
LORA_MODEL_DIR = "./final_lora_model"

def main():
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # We load the base model in its native bfloat16 to fit in VRAM and run fast
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True
    )

    print("Injecting fine-tuned adapter weights...")
    model = PeftModel.from_pretrained(model, LORA_MODEL_DIR)
    model.eval()

    # The prompt MUST perfectly match the format used during training
    instruction = "Write Python assembly bytecode to solve the following problem:"
    problem_description = """Write a program which prints multiplication tables in the following format:

1x1=1
1x2=2
.
.
9x8=72
9x9=81

Input

No input.

Output

1x1=1
1x2=2
.
.
9x8=72
9x9=81"""

    prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{problem_description}

### Response:\n"""

    print("\nGenerating Assembly Bytecode... (this might take a few seconds)")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1500,  # Assembly texts can be quite long
            temperature=0.1,      # Keep it strict and deterministic
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    print("\n" + "="*50)
    print("                OUTPUT                ")
    print("="*50 + "\n")
    
    # Slice the output to only show the newly generated text (ignoring the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(generated_text)

if __name__ == "__main__":
    main()
