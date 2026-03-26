import subprocess
import yaml
import os

def run_train(config_path, output_dir, seed):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Update config for the specific run
    cfg['train']['seed'] = seed
    cfg['train']['output_dir'] = output_dir

    os.makedirs(output_dir, exist_ok=True)
    best_model = os.path.join(output_dir, "best_model.pth")
    latest_ckpt = os.path.join(output_dir, "latest.pth")
    if os.path.exists(best_model):
        print(f"Skipping completed run for seed {seed}: {best_model}")
        return

    # Save temporary config
    temp_config = os.path.join(output_dir, f"config_seed{seed}.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)
    
    print(f"Starting training with seed {seed}, output to {output_dir}")
    cmd = ["python", "-m", "src.train", "--config", temp_config]
    if os.path.exists(latest_ckpt):
        print(f"Resuming existing run for seed {seed}: {latest_ckpt}")
        cmd.extend(["--resume", latest_ckpt])
    subprocess.run(cmd, check=True)
    
    # Cleanup
    if os.path.exists(temp_config):
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
