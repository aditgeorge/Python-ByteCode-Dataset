import re
import io
import dis
import multiprocessing
from datasets import load_dataset
import os

# 1. Primary Pattern: Strict markdown blocks
PRIMARY_PATTERN = re.compile(r'\`\`\`python(.*?)\`\`\`', flags=re.DOTALL | re.IGNORECASE)

# 2. Fallback Pattern: Grabs everything after "Here is the implementation ... :"
# The .*? allows for variations like "Here is the implementation of the function:"
FALLBACK_PATTERN = re.compile(r'Here is the implementation.*?\:\s*(.*?)', flags=re.DOTALL | re.IGNORECASE)

def process_and_verify(example):
    extracted_code = ""
    compilation_status = ""
    error_message = ""
    bytecode_str = ""
    extracted_code = ""
    try:
        text = str(example.get('output', '')) 
        
        # --- EXTRACTION LOGIC ---
        match = PRIMARY_PATTERN.search(text)
        
        if match:
            extracted_code = match.group(1).strip()
        
        # --- COMPILATION LOGIC ---
        if extracted_code:
            try:
                # Compile to verify valid Python syntax
                compiled_code = compile(extracted_code, '<string>', 'exec')
                
                # Disassemble into a readable bytecode string
                bytecode_io = io.StringIO()
                dis.dis(compiled_code, file=bytecode_io)
                bytecode_str = bytecode_io.getvalue()
                
                compilation_status = "Success"
                
            except SyntaxError as e:
                compilation_status = "SyntaxError"
                error_message = str(e)
            except Exception as e:
                compilation_status = "Unexpected Error"
                error_message = str(e)
        else:
            compilation_status = "Extraction Error"
            error_message = "No Python code block found in the output."
            
    except Exception as e:
        compilation_status = f"System Error: {str(e)}"
        
    # Add the new columns
    example['python_output'] = extracted_code
    example['status'] = compilation_status
    example['compiled_bytecode'] = bytecode_str  
    example['extracted_code'] = extracted_code if extracted_code else "No match found"
    example['error_message'] = error_message
    
    return example

if __name__ == "__main__":

    HF_TOKEN = os.getenv("HF_TOKEN")
    
    # 1. Load dataset
    print("Loading dataset metadata...")
    dataset = load_dataset("jtatman/python-code-dataset-500k",
                            split="train",
                            # streaming=True,
                            # token=HF_TOKEN)
                            token=True)

    dataset = dataset
    
    dataset = dataset.add_column("serial_number", range(len(dataset)))

    num_cores = multiprocessing.cpu_count()
    print(f"Starting processing across {num_cores} cores...")

    # 3. Apply processing
    processed_dataset = dataset.map(
        process_and_verify,
        num_proc=num_cores,
        desc="Extracting, Compiling, and Disassembling Code"
    )

    # 4. Save the test dataset
    processed_dataset.save_to_disk('./full_test_dataset')
    
    # --- 5. Analytics & Rate Calculation ---
    total_records = len(processed_dataset)
    
    # Filter to find the successes
    success_dataset = processed_dataset.filter(
        lambda x: x['status'] == "Success", 
        num_proc=num_cores,
        desc="Calculating Success Rate"
    )

    success_dataset.save_to_disk('./clean_test_dataset')
    
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
    # if success_count > 0:
    #     first_success = success_dataset[0]
    #     print("\n--- SAMPLE BYTECODE FROM FIRST SUCCESS ---")
    #     # Print just the first 500 characters of bytecode so it doesn't flood your screen
    #     print(first_success['compiled_bytecode'][:500] + "\n...[truncated]...")