import subprocess
import yaml
import os
import time

def run_experiment(exp_id, config_mods):
    print(f"--- Starting Experiment {exp_id} ---")
    # Load base config
    with open("config/disaster.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Apply modifications
    for k, v in config_mods.items():
        if isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
            
    # Save config
    config_path = f"config_{exp_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
        
    # Run confirmation runs (2 runs for mAP_confirm)
    run_confirm_cmd = f"python run_confirm.py --config {config_path} --exp_id {exp_id}"
    # We need to modify run_confirm.py to accept these args
    subprocess.run(["python", "run_confirm.py", config_path, exp_id], check=True)
    
    # Calculate CRI
    # manage_cri.py should return the score
    result = subprocess.run(["python", "manage_cri.py", exp_id], capture_output=True, text=True)
    cri = float(result.stdout.strip().split(":")[-1])
    return cri

def main():
    history = []
    
    # Exp 001: Baseline
    cri_001 = run_experiment("exp001", {})
    history.append(cri_001)
    
    # Exp 002: Focal Loss (Logic to enable it needs to be in train.py)
    # cri_002 = run_experiment("exp002", {"model": {"loss": "focal"}})
    # history.append(cri_002)
    
    # ... logic for 3 rounds of no increase ...
    print(f"Optimization history: {history}")

if __name__ == "__main__":
    main()
