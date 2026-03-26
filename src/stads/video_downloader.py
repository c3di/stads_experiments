import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Constants
GITHUB_REPO = "bharadwajakarsh/stads_adaptive_sampler"
GITHUB_RELEASE_TAG = "microscope-video-data"

DEFAULT_SAVE_DIR = Path.home() / ".stads_data"

def get_github_token():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        sys.stderr.write("ERROR: GITHUB_TOKEN is not set.\n")
        sys.exit(1)
    return token

def download_github_release_asset(filename: str, dest_path: Path):
    token = get_github_token()
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/octet-stream",
    }

    # Step 1: Get list of assets from release
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{GITHUB_RELEASE_TAG}"
    response = requests.get(api_url, headers={"Authorization": f"token {token}"})
    response.raise_for_status()
    assets = response.json()["assets"]

    # Step 2: Find the correct asset
    asset = next((a for a in assets if a["name"] == filename), None)
    if not asset:
        raise FileNotFoundError(f"Asset '{filename}' not found in GitHub release '{GITHUB_RELEASE_TAG}'.")

    download_url = asset["url"]

    # Step 3: Stream download using authenticated API
    with requests.get(download_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dest_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

def download_video(filename: str, force: bool = False) -> Path:
    dest_path = DEFAULT_SAVE_DIR / filename
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

    if dest_path.exists() and not force:
        return dest_path

    print(f"[INFO] Downloading {filename} from GitHub release '{GITHUB_RELEASE_TAG}' ...")
    download_github_release_asset(filename, dest_path)
    return dest_path
