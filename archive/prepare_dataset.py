import os
import json
import glob

import re
import html

def clean_problem_description(html_text):
    # 1. Remove everything starting from "Template for" headers
    match = re.search(r'(?i)<h2[^>]*>\s*Template for', html_text)
    if match:
        html_text = html_text[:match.start()]
        
    # 2. Remove all HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)
    
    # 3. Unescape HTML entities (like &le; &lt; etc)
    text = html.unescape(text)
    
    # 4. Clean up extra whitespace and blank lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def prepare_dataset(problem_desc_dir, pyc_text_dir, output_jsonl):
    """
    Combines the problem description HTML and the generated pyc text into a jsonl
    dataset suitable for instruction fine-tuning.
    """
    print("Generating dataset: Problem Description -> Assembly Bytecode")
    
    dataset_entries = []
    
    # We will iterate over the generated text directory because it only contains 
    # files that successfully compiled and parsed.
    p_dirs = sorted([d for d in os.listdir(pyc_text_dir) if d.startswith('p') and os.path.isdir(os.path.join(pyc_text_dir, d))])
    
    for p_dir in p_dirs:
        txt_folder = os.path.join(pyc_text_dir, p_dir)
        
        # Locate the problem description HTML
        html_path = os.path.join(problem_desc_dir, f"{p_dir}.html")
        if not os.path.exists(html_path):
            continue
            
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            problem_text = clean_problem_description(f.read().strip())
                
        for txt_filename in os.listdir(txt_folder):
            if not txt_filename.endswith(".txt"):
                continue
                
            txt_path = os.path.join(txt_folder, txt_filename)
            
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                bytecode_text = f.read().strip()
                
            instruction = "Write Python assembly bytecode to solve the following problem:"
                
            dataset_entries.append({
                "instruction": instruction,
                "input": problem_text,
                "output": bytecode_text
            })
            
    print(f"Total valid pairs found: {len(dataset_entries)}")
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in dataset_entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Dataset saved to: {output_jsonl}")

if __name__ == "__main__":
    PROBLEM_DIR = r"E:\Project_CodeNet\problem_descriptions"
    TEXT_DIR = r"E:\Project_CodeNet\newdata_text"
    OUTPUT_FILE = r"E:\create-dataset\dataset.jsonl"
    
    prepare_dataset(PROBLEM_DIR, TEXT_DIR, OUTPUT_FILE)
