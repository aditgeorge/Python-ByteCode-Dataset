import struct
import marshal
import time
import dis
import sys

def dump_code_to_text(c_obj, f, indent=0):
    indent_str = " " * indent
    f.write(f"{indent_str}Code object: {c_obj.co_name}\n")
    f.write(f"{indent_str}  argcount: {getattr(c_obj, 'co_argcount', 0)}\n")
    f.write(f"{indent_str}  posonlyargcount: {getattr(c_obj, 'co_posonlyargcount', 0)}\n")
    f.write(f"{indent_str}  kwonlyargcount: {getattr(c_obj, 'co_kwonlyargcount', 0)}\n")
    f.write(f"{indent_str}  nlocals: {getattr(c_obj, 'co_nlocals', 0)}\n")
    f.write(f"{indent_str}  stacksize: {getattr(c_obj, 'co_stacksize', 0)}\n")
    f.write(f"{indent_str}  flags: {getattr(c_obj, 'co_flags', 0)}\n")
    f.write(f"{indent_str}  firstlineno: {getattr(c_obj, 'co_firstlineno', 1)}\n")
    f.write(f"{indent_str}  filename: {getattr(c_obj, 'co_filename', '<string>')}\n")
    f.write(f"{indent_str}  varnames: {getattr(c_obj, 'co_varnames', [])}\n")
    f.write(f"{indent_str}  cellvars: {getattr(c_obj, 'co_cellvars', [])}\n")
    f.write(f"{indent_str}  freevars: {getattr(c_obj, 'co_freevars', [])}\n")
    f.write(f"{indent_str}  Instructions:\n")
    
    for instr in dis.get_instructions(c_obj):
        lineno = str(instr.starts_line) if instr.starts_line else ""
        
        # Format the instruction output to text
        arg_str = str(instr.arg) if instr.arg is not None else ""
        if instr.argrepr:
            if arg_str:
                arg_display = f"{arg_str} ({instr.argrepr})"
            else:
                arg_display = instr.argrepr
        else:
            arg_display = arg_str
            
        f.write(f"{indent_str}    {lineno:>4} {instr.offset:>4} {instr.opname:<20} {arg_display}\n")
        
        # Recursively handle nested code objects
        if type(instr.argval).__name__ == 'code':
            f.write(f"\n{indent_str}    --- Nested Code ---\n")
            dump_code_to_text(instr.argval, f, indent + 4)
            f.write(f"{indent_str}    -------------------\n\n")

def parse_pyc_to_text(file_path, output_path):
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

    # 3. EXTRACT AND SAVE HUMAN-READABLE INSTRUCTIONS AS TEXT
    print("--- Disassembly ---")
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write("--- .pyc Header ---\n")
        out_file.write(f"Magic Number : {magic_bytes.hex()}\n")
        out_file.write(f"Timestamp    : {time.ctime(timestamp)}\n")
        out_file.write(f"Source Size  : {file_size} bytes\n\n")
        out_file.write("--- Disassembly ---\n")
        dump_code_to_text(code_obj, out_file)
        
    print(f"\nInstructions successfully saved to {output_path}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert a .pyc file to plain text instructions representation.")
    parser.add_argument("input_file", help="Path to the input .pyc file")
    parser.add_argument("-o", "--output", default="llm_training_data.txt", help="Path to the output TXT file (default: llm_training_data.txt)")
    args = parser.parse_args()
    
    try:
        parse_pyc_to_text(args.input_file, args.output)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
    except EOFError:
        print("Error: The .pyc file is corrupted or incomplete.")
