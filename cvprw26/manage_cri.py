import os
import json
import yaml
import re

def parse_last_metrics(output_dir):
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        return None
    import torch
    checkpoint = torch.load(best_model_path, map_location='cpu')
    metrics = checkpoint.get('metrics', {})
    # Map category APs
    # segm_AP_intact, segm_AP_damaged, segm_AP_destroyed
    ap_dict = {
        'mAP': metrics.get('segm_AP', 0.0),
        'intact': metrics.get('segm_AP_intact', 0.0),
        'damaged': metrics.get('segm_AP_damaged', 0.0),
        'destroyed': metrics.get('segm_AP_destroyed', 0.0)
    }
    return ap_dict

def calculate_cri(map_confirm, ap_dict, r=1):
    t_intact, t_damaged, t_destroyed = 0.4000, 0.2000, 0.2500
    
    term1 = 0.70 * (map_confirm / 0.3500)
    
    ratios = [
        ap_dict.get('intact', 0) / t_intact,
        ap_dict.get('damaged', 0) / t_damaged,
        ap_dict.get('destroyed', 0) / t_destroyed
    ]
    # Handle zero division or very small targets if necessary
    term2 = 0.20 * min(ratios)
    term3 = 0.10 * r
    
    cri = 100 * (term1 + term2 + term3)
    return cri

def update_docs(exp_id, cri, metrics):
    # Logic to update Training_Log.md
    print(f"Exp {exp_id} - CRI: {cri:.2f}")
    pass

import sys

def main():
    if len(sys.argv) > 1:
        exp_id = sys.argv[1]
    else:
        exp_id = "exp001"
        
    run1_dir = f"outputs/{exp_id}_run1/"
    run2_dir = f"outputs/{exp_id}_run2/"
    
    m1 = parse_last_metrics(run1_dir)
    m2 = parse_last_metrics(run2_dir)
    
    if m1 and m2:
        map_confirm = (m1['mAP'] + m2['mAP']) / 2
        # Use metrics from run1 for individual APs (or average them)
        ap_avg = {k: (m1[k] + m2[k])/2 for k in m1}
        cri = calculate_cri(map_confirm, ap_avg, r=1)
        print(f"Calculated CRI: {cri:.4f}")
    else:
        print("Waiting for training to complete...")

if __name__ == "__main__":
    main()
