import struct
import marshal
import time
import dis
import json
import sys

def extract_instructions(c_obj):
    code_dict = {
        "name": c_obj.co_name,
        "argcount": getattr(c_obj, "co_argcount", 0),
        "posonlyargcount": getattr(c_obj, "co_posonlyargcount", 0),
        "kwonlyargcount": getattr(c_obj, "co_kwonlyargcount", 0),
        "nlocals": getattr(c_obj, "co_nlocals", 0),
        "stacksize": getattr(c_obj, "co_stacksize", 0),
        "flags": getattr(c_obj, "co_flags", 0),
        "firstlineno": getattr(c_obj, "co_firstlineno", 1),
        "filename": getattr(c_obj, "co_filename", "<string>"),
        "varnames": list(getattr(c_obj, "co_varnames", [])),
        "cellvars": list(getattr(c_obj, "co_cellvars", [])),
        "freevars": list(getattr(c_obj, "co_freevars", [])),
        "instructions": []
    }
    
    current_lineno = getattr(c_obj, "co_firstlineno", 1)
    
    for instr in dis.get_instructions(c_obj):
        lineno_to_save = getattr(instr, 'starts_line', None)
            
        argval = instr.argval
        is_code = False
        code_data = None
        target_offset = None
        
        if type(argval).__name__ == 'code':
            is_code = True
            code_data = extract_instructions(argval)
            val_to_save = None
        elif 'JUMP' in instr.opname or instr.opname == 'FOR_ITER' or instr.opname == 'SETUP_FINALLY':
            if isinstance(argval, int):
                target_offset = argval
            val_to_save = None
        else:
            val_to_save = instr.argrepr

        code_dict["instructions"].append({
            "offset": instr.offset,
            "opname": instr.opname,
            "arg": instr.arg,
            "argrepr": instr.argrepr,
            "argval_saved": val_to_save,
            "is_code": is_code,
            "code_data": code_data,
            "target_offset": target_offset,
            "lineno": lineno_to_save
        })
        
    return code_dict

def parse_pyc(file_path):
    print(f"Parsing: {file_path}\n")
    
    with open(file_path, 'rb') as f:
        # 1. PARSE THE 16-BYTE HEADER (Python 3.7+)
        header_bytes = f.read(16)
        magic_bytes = header_bytes[:4]
        bitfield = struct.unpack('<I', header_bytes[4:8])[0]
        timestamp = struct.unpack('<I', header_bytes[8:12])[0]
        file_size = struct.unpack('<I', header_bytes[12:16])[0]

        print("--- .pyc Header ---")
        print(f"Magic Number : {magic_bytes.hex()}")
        print(f"Timestamp    : {time.ctime(timestamp)}")
        print(f"Source Size  : {file_size} bytes\n")

        # 2. LOAD THE RAW CODE OBJECT
        code_obj = marshal.load(f)

    # 3. EXTRACT HUMAN-READABLE INSTRUCTIONS
    print("--- Disassembly ---")
    
    dataset = {
        "header_hex": header_bytes.hex(),
        "code": extract_instructions(code_obj)
    }
    
    return dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a .pyc file to a hierarchical JSON representation.")
    parser.add_argument("input_file", help="Path to the input .pyc file")
    parser.add_argument("-o", "--output", default="llm_training_data.json", help="Path to the output JSON file (default: llm_training_data.json)")
    args = parser.parse_args()
    
    try:
        dataset = parse_pyc(args.input_file)
        
        with open(args.output, "w") as out_file:
            json.dump(dataset, out_file, indent=4)
            
        print(f"\nDataset successfully saved to {args.output}!")
        
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
    except EOFError:
        print("Error: The .pyc file is corrupted or incomplete.")