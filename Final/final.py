import re
import io
import dis
import multiprocessing
from datasets import load_dataset

# Pre-compile regex
PATTERN = re.compile(r'```(?:python)?\s*(.*?)\s*```', flags=re.DOTALL | re.IGNORECASE)

def process_and_verify(example):
    extracted_code = ""
    compilation_status = ""
    bytecode_str = ""
    
    try:
        text = str(example.get('output', '')) 
        match = PATTERN.search(text)
        
        if match:
            extracted_code = match.group(1).strip()
            
            try:
                # 1. Compile the code into an Abstract Syntax Tree / code object
                compiled_code = compile(extracted_code, '<string>', 'exec')
                
                # 2. Disassemble the code object into a readable bytecode string
                bytecode_io = io.StringIO()
                dis.dis(compiled_code, file=bytecode_io)
                bytecode_str = bytecode_io.getvalue()
                
                compilation_status = "Success"
                
            except SyntaxError as e:
                compilation_status = f"SyntaxError: {str(e)}"
            except Exception as e:
                compilation_status = f"Unexpected Error: {str(e)}"
        else:
            compilation_status = "Extraction Error: No markdown code block found"
            
    except Exception as e:
        compilation_status = f"System Error: {str(e)}"
        
    # Add the new columns
    example['python_output'] = extracted_code
    example['compiled_output'] = compilation_status
    example['compiled_bytecode'] = bytecode_str  
    
    return example

if __name__ == "__main__":
    # Insert your token or set to True if using huggingface-cli login
    HF_TOKEN = True  # or "hf_your_actual_token_string_here"
    
    # 1. Load dataset
    print("Loading dataset metadata...")
    dataset = load_dataset('your_dataset_name', split='train', token=HF_TOKEN)
    
    # --- 2. TEST BATCH: Select only the first 50 records ---
    dataset = dataset.select(range(50))
    print("Testing on 50 records...")
    
    num_cores = multiprocessing.cpu_count()
    print(f"Starting processing across {num_cores} cores...")

    # 3. Apply processing
    processed_dataset = dataset.map(
        process_and_verify,
        num_proc=num_cores,
        desc="Extracting, Compiling, and Disassembling Code"
    )

    # 4. Save the test dataset
    processed_dataset.save_to_disk('./test_processed_code_dataset')
    
    # --- 5. Analytics & Rate Calculation ---
    total_records = len(processed_dataset)
    
    # Filter to find the successes
    success_dataset = processed_dataset.filter(
        lambda x: x['compiled_output'] == "Success", 
        num_proc=num_cores,
        desc="Calculating Success Rate"
    )
    
    success_count = len(success_dataset)
    failure_count = total_records - success_count
    
    success_rate = (success_count / total_records * 100) if total_records > 0 else 0
    failure_rate = (failure_count / total_records * 100) if total_records > 0 else 0
    
    print("\n" + "="*40)
    print("      DATASET PROCESSING REPORT")
    print("="*40)
    print(f"Total Records Processed : {total_records:,}")
    print(f"Successful Extractions  : {success_count:,} ({success_rate:.2f}%)")
    print(f"Failed Extractions      : {failure_count:,} ({failure_rate:.2f}%)")
    print("="*40)
    
    # Print the first successful row to visually verify the bytecode
    if success_count > 0:
        first_success = success_dataset[0]
        print("\n--- SAMPLE BYTECODE FROM FIRST SUCCESS ---")
        # Print just the first 500 characters of bytecode so it doesn't flood your screen
        print(first_success['compiled_bytecode'][:500] + "\n...[truncated]...")