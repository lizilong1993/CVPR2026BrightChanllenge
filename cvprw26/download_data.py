import os
import requests
from tqdm import tqdm
import zipfile
import time

# Zenodo record ID
RECORD_ID = "14619797"
FILES_TO_DOWNLOAD = [
    "pre-event.zip",
    "post-event.zip",
    "target.zip",
    "cvprw26_trainval_instance_labels.zip"
]

def get_json_with_retry(url, max_retries=5):
    import subprocess
    import json
    for i in range(max_retries):
        try:
            cmd = ["curl.exe", "-L", "-s", url]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            if i < max_retries - 1:
                time.sleep(2 ** i)
            else:
                raise

def download_file(url, filename):
    print(f"Downloading {filename}...")
    import subprocess
    cmd = ["curl.exe", "-L", "-o", filename, url]
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Download failed for {filename}: {e}")
        raise

def unzip_file(filename, extract_to):
    print(f"Unzipping {filename} to {extract_to}...")
    import zipfile
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    import time
    # Get Zenodo record info
    api_url = f"https://zenodo.org/api/records/{RECORD_ID}"
    data = get_json_with_retry(api_url)
    
    if not os.path.exists("data"):
        os.makedirs("data")
    
    bright_root = "BRIGHT_DATA"
    if not os.path.exists(bright_root):
        os.makedirs(bright_root)

    for file_info in data['files']:
        filename = file_info['key']
        if filename in FILES_TO_DOWNLOAD:
            download_url = file_info['links']['self']
            dest_path = os.path.join(bright_root, filename)
            
            if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
                download_file(download_url, dest_path)
            else:
                print(f"{filename} already exists, skipping download.")
            
            # Special handling for target.zip to target_instance_level
            extract_to = bright_root
            if filename == "target.zip":
                extract_to = os.path.join(bright_root, "target_instance_level")
            
            # Unzip
            unzip_file(dest_path, extract_to)
            print(f"Extraction for {filename} done.")

if __name__ == "__main__":
    main()
