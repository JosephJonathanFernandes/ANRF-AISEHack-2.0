import json
import sys
import os

def upgrade_notebook(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    imports_injected = False
    models_injected = False
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        # 1. Inject imports
        source = cell['source']
        if not imports_injected and any("import lightgbm" in line for line in source):
            print("Found imports cell, injecting MLPRegressor and SVR imports...")
            # Insert at the end of the imports
            cell['source'].append("from sklearn.neural_network import MLPRegressor\n")
            cell['source'].append("from sklearn.svm import SVR\n")
            imports_injected = True
            
        # 2. Inject models into stage_b_models
        if not models_injected and any("def stage_b_models" in line for line in source):
            print("Found stage_b_models cell, injecting new models...")
            new_source = []
            for line in source:
                new_source.append(line)
                if '"ridge": lambda: make_pipeline(' in line:
                    new_source.append('        "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64), learning_rate_init=0.005, max_iter=400, early_stopping=True, random_state=SEED)),\n')
                    new_source.append('        "svr": lambda: make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1)),\n')
                    models_injected = True
            cell['source'] = new_source
            
    if not imports_injected:
        print("Warning: Could not find imports block to inject sklearn models!")
    if not models_injected:
        print("Warning: Could not find stage_b_models to inject new models!")
        
    print(f"Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Done!")

if __name__ == "__main__":
    os.makedirs(r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\12", exist_ok=True)
    
    input_file = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\Shivesh\anrf-app3.ipynb"
    output_file = r"C:\Users\Joseph\Desktop\projects\ANRF-AISEHack-2.0\submission_R2\12\anrf-app5-v12.ipynb"
    upgrade_notebook(input_file, output_file)
