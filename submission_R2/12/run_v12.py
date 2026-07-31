import json
import sys
import tempfile
import os
import subprocess

def run_notebook(notebook_path):
    print(f"Extracting code from {notebook_path}...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Skip cells that only contain jupyter magics like !pip install
            source = [line for line in cell['source'] if not line.strip().startswith('!')]
            code_cells.append("".join(source))
            
    full_script = "\n\n".join(code_cells)
    
    script_path = notebook_path.replace(".ipynb", "_extracted.py")
    print(f"Writing extracted Python script to {script_path}...")
    with open(script_path, "w", encoding='utf-8') as f:
        f.write(full_script)
        
    print(f"Running script...")
    # Change to the project root so the script can find the dataset folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(script_path), "..", ".."))
    subprocess.run([sys.executable, script_path], cwd=project_root)
    print("Done!")

if __name__ == "__main__":
    notebook_file = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\12\anrf-app5-v12.ipynb"
    run_notebook(notebook_file)
