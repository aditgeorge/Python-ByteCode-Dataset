from datasets import load_from_disk

# 2. Define the Chain of Thought formatting template
def format_cot_prompt(example):
    instruction = example['instruction'] 
    python_code = example['python_output']
    bytecode = example['compiled_bytecode']
    
    # Structure the response with the hidden scratchpad block
    text = f"""Below is an instruction that describes a programming task.

### Instruction:
{instruction}

### Response:
<scratchpad>
To ensure accurate bytecode generation, I will write the Python implementation first.

```python
{python_code}
```

Now, I will translate this Python logic into stack-based bytecode operations.
```
{bytecode}
```"""
    
    # SFTTrainer expects a "text" column
    return {"text": text}

if __name__ == "__main__":
    # 1. Load the dataset from disk
    temp_dataset = load_from_disk('./clean_test_dataset')
    print(f"Loaded {len(temp_dataset)} records. Starting formatting...")

    # 3. Map the formatting function across the entire dataset
    # num_proc=8 runs this in parallel across 8 CPU cores for massive speed
    formatted_dataset = temp_dataset.map(format_cot_prompt, num_proc=8)

    # 4. Save the new, training-ready dataset to disk
    formatted_dataset.save_to_disk('./final_training_dataset')

    print("Successfully formatted and saved!")

    # 5. Print a quick preview to verify it looks perfect
    print("\n--- PREVIEW OF RECORD 1 ---")
    print(formatted_dataset[0]['text'])