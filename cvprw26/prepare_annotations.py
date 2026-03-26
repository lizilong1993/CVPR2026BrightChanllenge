import subprocess
import os
import sys

def main():
    # Paths according to README and disaster.yaml
    bright_root = "BRIGHT_DATA"
    target_dir = os.path.join(bright_root, "target_instance_level")
    post_event_dir = os.path.join(bright_root, "post-event")
    pre_event_dir = os.path.join(bright_root, "pre-event")
    splits_dir = "data/splits"
    output_dir = "data/instance_annotations"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if data exists
    if not os.path.exists(target_dir) or not os.path.exists(post_event_dir) or not os.path.exists(pre_event_dir):
        print(f"Error: Data not found in {bright_root}. Please run download_data.py and unzip the files first.")
        print(f"Expected directories:\n - {target_dir}\n - {post_event_dir}\n - {pre_event_dir}")
        return

    print("Merging COCO JSON annotations...")
    cmd = [
        sys.executable, "tools/merge_coco_json.py",
        "--json-dir", target_dir,
        "--image-dir", post_event_dir,
        "--pre-event-dir", pre_event_dir,
        "--splits-dir", splits_dir,
        "--output-dir", output_dir
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Annotations prepared successfully in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running merge_coco_json.py: {e}")

if __name__ == "__main__":
    main()
