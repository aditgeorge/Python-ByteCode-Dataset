import os
import glob
import py_compile
import io
import dis
import marshal
import sys
import contextlib

# Import our custom serialization modules
from pyc_to_text import parse_pyc_to_text
from text_to_pyc import parse_text_to_dict, assemble_dict_to_bytecode

import re

def get_dis_tokens(co):
    out = io.StringIO()
    dis.dis(co, file=out)
    output = out.getvalue()
    # Strip memory addresses to allow stable comparison
    output = re.sub(r'at 0x[0-9A-Fa-f]+', 'at 0xXXXX', output)
    # Strip instruction index/arg when argrepr is present in parentheses
    output = re.sub(r'\s+\d+\s+\(', ' (', output)
    # Strip line numbers
    output = re.sub(r'(?m)^\s*\d+(?=\s+(?:\>\>\s+)?\d+\s+[A-Z_]+)', '', output)
    
    tokens = output.split()
    return [t for t in tokens if t != '>>' and not t.isdigit()]

def process_file(source_py, target_txt_path):
    temp_pyc = source_py + ".pyc"
    try:
        # 1. Compile to pyc
        py_compile.compile(source_py, cfile=temp_pyc, doraise=True)
        
        # 2. Parse pyc to Text dataset
        os.makedirs(os.path.dirname(target_txt_path), exist_ok=True)
        parse_pyc_to_text(temp_pyc, target_txt_path)
        
        # 3. Read back text to dict, then assemble Bytecode
        dataset = parse_text_to_dict(target_txt_path)
        code_dict = dataset["code"]
        bc = assemble_dict_to_bytecode(code_dict)
        new_co = bc.to_code()
        
        # 4. Verify against original pyc
        with open(temp_pyc, 'rb') as f:
            f.read(16) # Skip header
            orig_co = marshal.load(f)
            
        orig_dis = get_dis_tokens(orig_co)
        new_dis = get_dis_tokens(new_co)
        
        if orig_dis != new_dis:
            # Clean up the text file if disassembly mismatch
            if os.path.exists(target_txt_path):
                os.remove(target_txt_path)
            return False, "Disassembly mismatch"
            
        return True, "Success"

    except py_compile.PyCompileError as e:
        return False, "Syntax/Compile Error"
    except Exception as e:
        # Clean up the text file if exception
        if os.path.exists(target_txt_path):
            try:
                os.remove(target_txt_path)
            except OSError:
                pass
        return False, f"Exception: {type(e).__name__} {e}"
    finally:
        if os.path.exists(temp_pyc):
            try:
                os.remove(temp_pyc)
            except OSError:
                pass

def main():
    source_dir = r"E:\Project_CodeNet\data"
    target_dir = r"E:\Project_CodeNet\newdata_text"
    
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # For testing purposes, limit execution to 50 files. 
    # Change to None to process the entire dataset.
    MAX_FILES = 100
    
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
                
        dir_successes = 0
                
        for filename in os.listdir(py_folder):
            if filename.endswith(".py"):
                if MAX_FILES and processed >= MAX_FILES:
                    print(f"\nReached test limit of {MAX_FILES} files. Stopping.")
                    print(f"Summary: Processed {processed}, Success: {successes}, Failures: {failures}")
                    return

                source_py = os.path.join(py_folder, filename)
                txt_filename = filename.replace(".py", ".txt")
                target_txt_path = os.path.join(target_dir, p_dir, txt_filename)
                
                print(f"Processing {p_dir}/{filename}...", end=" ", flush=True)
                
                # Suppress prints from pyc_to_text to keep terminal clean
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    success, msg = process_file(source_py, target_txt_path)
                
                if success:
                    print("OK")
                    successes += 1
                    dir_successes += 1
                else:
                    print(f"FAIL ({msg})")
                    failures += 1
                
                processed += 1
                
                if dir_successes >= 3:
                    print(f"  -> Found 3 successful files in {p_dir}, moving to next folder.")
                    break
                
    print(f"\nFinished. Summary: Processed {processed}, Success: {successes}, Failures: {failures}")

if __name__ == "__main__":
    main()
