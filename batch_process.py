import os
import glob
import py_compile
import json
import io
import dis
import marshal
import sys
import contextlib

# Import our custom serialization modules
from pyc_to_json import parse_pyc
from json_to_pyc import assemble_dict_to_bytecode

import re

def get_dis_output(co):
    out = io.StringIO()
    dis.dis(co, file=out)
    output = out.getvalue()
    # Strip memory addresses to allow stable comparison
    output = re.sub(r'at 0x[0-9A-Fa-f]+', 'at 0xXXXX', output)
    # Strip instruction index/arg when argrepr is present in parentheses
    output = re.sub(r'\s+\d+\s+\(', ' (', output)
    return output

def process_file(source_py, target_json_path):
    temp_pyc = source_py + ".pyc"
    try:
        # 1. Compile to pyc
        py_compile.compile(source_py, cfile=temp_pyc, doraise=True)
        
        # 2. Parse pyc to JSON dataset
        dataset = parse_pyc(temp_pyc)
        
        # 3. Assemble JSON dataset back to Bytecode
        code_dict = dataset["code"]
        bc = assemble_dict_to_bytecode(code_dict)
        new_co = bc.to_code()
        
        # 4. Verify against original pyc
        with open(temp_pyc, 'rb') as f:
            f.read(16) # Skip header
            orig_co = marshal.load(f)
            
        orig_dis = get_dis_output(orig_co)
        new_dis = get_dis_output(new_co)
        
        if orig_dis != new_dis:
            return False, "Disassembly mismatch"
            
        # 5. Save JSON
        os.makedirs(os.path.dirname(target_json_path), exist_ok=True)
        with open(target_json_path, 'w') as f:
            json.dump(dataset, f, indent=4)
            
        return True, "Success"

    except py_compile.PyCompileError as e:
        return False, "Syntax/Compile Error"
    except Exception as e:
        return False, f"Exception: {type(e).__name__} {e}"
    finally:
        if os.path.exists(temp_pyc):
            try:
                os.remove(temp_pyc)
            except OSError:
                pass

def main():
    source_dir = r"E:\Project_CodeNet\data"
    target_dir = r"E:\Project_CodeNet\newdata"
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # For testing purposes, limit execution to 50 files. 
    # Change to None to process the entire dataset.
    MAX_FILES = 50 
    
    processed = 0
    successes = 0
    failures = 0
    
    print(f"Starting batch process. Source: {source_dir}")
    
    p_dirs = sorted([d for d in os.listdir(source_dir) if d.startswith('p') and os.path.isdir(os.path.join(source_dir, d))])
    
    for p_dir in p_dirs:
        py_folder = os.path.join(source_dir, p_dir, "Python")
        if not os.path.exists(py_folder):
            py_folder = os.path.join(source_dir, p_dir, "python")
            if not os.path.exists(py_folder):
                continue
                
        for filename in os.listdir(py_folder):
            if filename.endswith(".py"):
                if MAX_FILES and processed >= MAX_FILES:
                    print(f"\nReached test limit of {MAX_FILES} files. Stopping.")
                    print(f"Summary: Processed {processed}, Success: {successes}, Failures: {failures}")
                    return

                source_py = os.path.join(py_folder, filename)
                json_filename = filename.replace(".py", ".json")
                target_json_path = os.path.join(target_dir, p_dir, json_filename)
                
                print(f"Processing {p_dir}/{filename}...", end=" ", flush=True)
                
                # Suppress prints from pyc_to_json to keep terminal clean
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    success, msg = process_file(source_py, target_json_path)
                
                if success:
                    print("OK")
                    successes += 1
                else:
                    print(f"FAIL ({msg})")
                    failures += 1
                
                processed += 1
                
    print(f"\nFinished. Summary: Processed {processed}, Success: {successes}, Failures: {failures}")

if __name__ == "__main__":
    main()
