import json
import marshal
import ast
from bytecode import Bytecode, Instr, Label
from bytecode.flags import CompilerFlags

def assemble_dict_to_bytecode(code_dict):
    bc = Bytecode()
    bc.name = code_dict.get("name", "<module>")
    bc.argcount = code_dict.get("argcount", 0)
    bc.posonlyargcount = code_dict.get("posonlyargcount", 0)
    bc.kwonlyargcount = code_dict.get("kwonlyargcount", 0)
    
    flags_val = code_dict.get("flags", 0)
    # Convert integer flags back to CompilerFlags
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
        argrepr = item['argrepr']
        raw_arg = item['arg']
        is_code = item.get("is_code", False)
        code_data = item.get("code_data")
        target_offset = item.get("target_offset")
        argval_saved = item.get("argval_saved")

        # Insert Label if this offset is a jump destination
        if offset in label_map:
            bc.append(label_map[offset])

        has_arg = (raw_arg is not None)
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
                from bytecode import CellVar, FreeVar
                if argrepr in cellvars:
                    arg = CellVar(argrepr)
                else:
                    arg = FreeVar(argrepr)
                
            elif opname == 'COMPARE_OP':
                from bytecode import Compare
                arg = Compare(int(raw_arg))
                
            else:
                try:
                    arg = int(raw_arg)
                except (ValueError, TypeError):
                    arg = raw_arg

        lineno = item.get("lineno")

        if has_arg:
            bc.append(Instr(opname, arg, lineno=lineno))
        else:
            bc.append(Instr(opname, lineno=lineno))

    return bc

def assemble_json_to_pyc(json_path, output_pyc_path):
    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    header_bytes = bytes.fromhex(dataset["header_hex"])
    code_dict = dataset["code"]

    print("Compiling bytecode...")
    bc = assemble_dict_to_bytecode(code_dict)
    code_obj = bc.to_code()

    with open(output_pyc_path, 'wb') as f:
        f.write(header_bytes)
        marshal.dump(code_obj, f)

    print("Successfully assembled!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Assemble a hierarchical JSON representation back into a .pyc file.")
    parser.add_argument("input_file", help="Path to the input .json file")
    parser.add_argument("-o", "--output", default="llm_generated.pyc", help="Path to the output .pyc file (default: llm_generated.pyc)")
    args = parser.parse_args()
    
    try:
        assemble_json_to_pyc(args.input_file, args.output)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}")
    except Exception as e:
        print(f"Error: Failed to assemble pyc. Details: {e}")