import subprocess
import yaml
import os
import copy

def run_train(config_path, output_dir, seed):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Update config for the specific run
    cfg['train']['seed'] = seed
    cfg['train']['output_dir'] = output_dir
    
    # Save temporary config
    temp_config = f"config_temp_{seed}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"Starting training with seed {seed}, output to {output_dir}")
    cmd = ["python", "-m", "src.train", "--config", temp_config]
    subprocess.run(cmd, check=True)
    
    # Cleanup
    os.remove(temp_config)

import sys

def main():
    if len(sys.argv) > 2:
        config_path = sys.argv[1]
        exp_id = sys.argv[2]
    else:
        config_path = "config/disaster.yaml"
        exp_id = "exp001"
    
    # Run 1
    run_train(config_path, f"outputs/{exp_id}_run1/", 42)
    
    # Run 2
    run_train(config_path, f"outputs/{exp_id}_run2/", 123)
    
    # Calculate CRI
    print(f"All runs for {exp_id} completed. Calculating CRI...")
    subprocess.run(["python", "manage_cri.py", exp_id])

if __name__ == "__main__":
    main()
