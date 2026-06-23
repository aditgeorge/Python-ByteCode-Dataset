import sys
import time
import ast
import struct
import marshal
import dis

from bytecode import Bytecode, Instr, Label, CellVar, FreeVar, Compare
from bytecode.flags import CompilerFlags

def parse_text_to_dict(text_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    index = 0
    header_data = {}
    
    # Parse Header
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if line == "--- Disassembly ---":
            break
        if line.startswith("Magic Number : "):
            header_data['magic'] = bytes.fromhex(line.split(" : ")[1])
        elif line.startswith("Timestamp    : "):
            pass # We generate a new timestamp below
        elif line.startswith("Source Size  : "):
            header_data['size'] = int(line.split(" : ")[1].replace(" bytes", ""))
            
    # Parse code object
    code_dict, _ = parse_code_object(lines, index, 0)
    
    dataset = {
        "header_magic": header_data.get('magic', b'\x55\x0d\x0d\x0a'),
        "header_size": header_data.get('size', 0),
        "code": code_dict
    }
    return dataset

def parse_code_object(lines, index, indent_level):
    code_dict = {
        "instructions": []
    }
    indent_str = " " * indent_level
    
    # Parse attributes
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        
        if not line.startswith(indent_str) and stripped != "":
            break
            
        if stripped == "--- Nested Code ---" or stripped == "-------------------":
            index += 1
            continue
            
        if stripped.startswith("Code object: "):
            code_dict["name"] = stripped.split(": ", 1)[1]
            index += 1
            continue
            
        if ":" in stripped and not stripped.startswith("Instructions:"):
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in ["argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "firstlineno"]:
                code_dict[key] = int(val)
            elif key == "filename":
                code_dict[key] = val
            elif key in ["varnames", "cellvars", "freevars"]:
                try:
                    code_dict[key] = ast.literal_eval(val)
                except Exception:
                    code_dict[key] = []
            index += 1
            continue
            
        if stripped == "Instructions:":
            index += 1
            break
            
        index += 1

    # Parse instructions
    instr_indent = indent_str + "    "
    last_lineno = 1
    if "firstlineno" in code_dict:
        last_lineno = code_dict["firstlineno"]
        
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
            
        if not line.startswith(instr_indent):
            break
            
        stripped_full = line[len(instr_indent):]
        stripped = stripped_full.strip()
        
        if stripped == "--- Nested Code ---":
            index += 1
            nested_code_dict, index = parse_code_object(lines, index, indent_level + 4)
            # Find closing
            while index < len(lines) and lines[index].strip() != "-------------------":
                index += 1
            if index < len(lines):
                index += 1
                
            if code_dict["instructions"]:
                last_instr = code_dict["instructions"][-1]
                last_instr["is_code"] = True
                last_instr["code_data"] = nested_code_dict
            continue
            
        # Example line: "   1    0 LOAD_CONST           0 (<code object ...>)"
        if len(stripped_full) < 30:
            index += 1
            continue
            
        lineno_str = stripped_full[0:4].strip()
        if lineno_str:
            try:
                lineno = int(lineno_str)
                last_lineno = lineno
            except ValueError:
                lineno = last_lineno
        else:
            lineno = last_lineno
            
        offset_str = stripped_full[5:9].strip()
        try:
            offset = int(offset_str)
        except ValueError:
            offset = 0
            
        opname = stripped_full[10:30].strip()
        arg_display = stripped_full[31:].strip()
        
        arg = None
        argrepr = arg_display
        if arg_display:
            parts = arg_display.split(" ", 1)
            if parts[0].isdigit() or (parts[0].startswith('-') and parts[0][1:].isdigit()):
                arg_val_str = parts[0]
                if len(parts) > 1 and parts[1].startswith('(') and parts[1].endswith(')'):
                    arg = int(arg_val_str)
                    argrepr = parts[1][1:-1]
                elif len(parts) == 1:
                    arg = int(arg_val_str)
                    argrepr = ""
            
        instr_data = {
            "lineno": lineno,
            "offset": offset,
            "opname": opname,
            "arg": arg,
            "argrepr": argrepr,
            "is_code": False,
            "code_data": None
        }
        
        # Infer target_offset for jump operations
        if 'JUMP' in opname or opname == 'FOR_ITER' or opname == 'SETUP_FINALLY':
            target_str = argrepr.replace("to ", "").strip()
            try:
                instr_data["target_offset"] = int(target_str)
            except ValueError:
                pass
                
        code_dict["instructions"].append(instr_data)
        index += 1
        
    return code_dict, index

def assemble_dict_to_bytecode(code_dict):
    bc = Bytecode()
    bc.name = code_dict.get("name", "<module>")
    bc.argcount = code_dict.get("argcount", 0)
    bc.posonlyargcount = code_dict.get("posonlyargcount", 0)
    bc.kwonlyargcount = code_dict.get("kwonlyargcount", 0)
    
    flags_val = code_dict.get("flags", 0)
    bc.flags = CompilerFlags(flags_val)
    
    bc.first_lineno = code_dict.get("firstlineno", 1)
    bc.filename = code_dict.get("filename", "<string>")
    
    cellvars_list = list(code_dict.get("cellvars", []))
    freevars_list = list(code_dict.get("freevars", []))
    varnames = list(code_dict.get("varnames", []))
    
    total_args = bc.argcount + bc.posonlyargcount + bc.kwonlyargcount
    bc.argnames = varnames[:total_args]
    bc.cellvars = cellvars_list
    bc.freevars = freevars_list
    
    cellvars = set(cellvars_list)
    freevars = set(freevars_list)
    
    instructions_json = code_dict["instructions"]
    
    # 1. FIRST PASS: MAP JUMP TARGETS TO LABELS
    jump_targets = set()
    for item in instructions_json:
        if item.get("target_offset") is not None:
            jump_targets.add(item["target_offset"])

    label_map = {offset: Label() for offset in jump_targets}
    
    # 2. SECOND PASS: BUILD THE BYTECODE OBJECT
    for item in instructions_json:
        offset = item['offset']
        opname = item['opname']
        raw_arg = item.get('arg')
        argrepr = item['argrepr']
        is_code = item.get("is_code", False)
        code_data = item.get("code_data")
        target_offset = item.get("target_offset")

        if offset in label_map:
            bc.append(label_map[offset])

        try:
            opcode = dis.opname.index(opname)
            has_arg = opcode >= dis.HAVE_ARGUMENT
        except ValueError:
            has_arg = False

        arg = None
        
        if has_arg:
            if target_offset is not None:
                arg = label_map[target_offset]
                
            elif opname == 'LOAD_CONST':
                if is_code:
                    arg = assemble_dict_to_bytecode(code_data).to_code()
                elif argrepr == 'None':
                    arg = None
                else:
                    try:
                        arg = ast.literal_eval(argrepr)
                    except Exception:
                        arg = argrepr
                        
            elif opname in ['LOAD_NAME', 'STORE_NAME', 'LOAD_FAST', 'STORE_FAST', 'LOAD_GLOBAL', 'STORE_GLOBAL', 'LOAD_ATTR', 'STORE_ATTR', 'LOAD_METHOD', 'IMPORT_NAME', 'IMPORT_FROM', 'DELETE_FAST', 'DELETE_NAME', 'DELETE_GLOBAL']:
                arg = argrepr
                
            elif opname in ['LOAD_DEREF', 'STORE_DEREF', 'DELETE_DEREF', 'LOAD_CLOSURE', 'LOAD_CLASSFREE']:
                if argrepr in cellvars:
                    arg = CellVar(argrepr)
                else:
                    arg = FreeVar(argrepr)
                
            elif opname == 'COMPARE_OP':
                try:
                    op_idx = dis.cmp_op.index(argrepr)
                    arg = Compare(op_idx)
                except ValueError:
                    arg = Compare(raw_arg if raw_arg is not None else 0)
                
            else:
                if raw_arg is not None:
                    arg = raw_arg
                else:
                    try:
                        arg = int(argrepr)
                    except (ValueError, TypeError):
                        arg = argrepr

        lineno = item.get("lineno")

        if has_arg:
            bc.append(Instr(opname, arg, lineno=lineno))
        else:
            bc.append(Instr(opname, lineno=lineno))

    return bc

def assemble_text_to_pyc(text_path, output_pyc_path):
    print(f"Loading dataset from: {text_path}")
    dataset = parse_text_to_dict(text_path)

    magic = dataset.get("header_magic", b'\x55\x0d\x0d\x0a')
    bitfield = b'\x00\x00\x00\x00'
    timestamp = struct.pack('<I', int(time.time()))
    size = struct.pack('<I', dataset.get("header_size", 0))
    header_bytes = magic + bitfield + timestamp + size

    code_dict = dataset["code"]

    print("Compiling bytecode...")
    bc = assemble_dict_to_bytecode(code_dict)
    code_obj = bc.to_code()

    with open(output_pyc_path, 'wb') as f:
        f.write(header_bytes)
        marshal.dump(code_obj, f)

    print(f"Successfully assembled to {output_pyc_path}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assemble a plain text instructions representation back into a .pyc file.")
    parser.add_argument("input_file", help="Path to the input .txt file")
    parser.add_argument("-o", "--output", default="llm_generated_from_txt.pyc", help="Path to the output .pyc file (default: llm_generated_from_txt.pyc)")
    args = parser.parse_args()
    
    try:
        assemble_text_to_pyc(args.input_file, args.output)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
    except Exception as e:
        print(f"Error: Failed to assemble pyc. Details: {e}")
