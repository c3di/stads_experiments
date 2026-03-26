import sys
from .read_images import get_frames_from_tif
from .video_downloader import download_video

SIGMA = 4
KERNEL_SIZE = int(SIGMA * 3) + 1

# Define videos and their frame limits
VIDEOS = {
    "Hydration.tif": 44,
    "Li_Expulsion_1.tif": 20,
    "Li_Expulsion_2.tif": 22,
    "Si_Lithiation.tif": 5,
}

LOADED = {}

def safe_download(video_name: str):
    try:
        sourcePath = download_video(video_name)
        return sourcePath
    except Exception as e:
        print(f"[ERROR] Failed to download {video_name}: {e}", file=sys.stderr)
        return None

for video_filename, frame_limit in VIDEOS.items():
    key = video_filename.split(".")[0].upper()
    path = safe_download(video_filename)
    if path:
        LOADED[key] = get_frames_from_tif(str(path), frame_limit)
    else:
        LOADED[key] = None
        print(f"[WARN] {key} is not available.", file=sys.stderr)

# Raise error if all videos failed
if all(v is None for v in LOADED.values()):
    raise RuntimeError("None of the videos could be downloaded — check GITHUB_TOKEN or internet access.")

# Optional: expose individual variables
HYDRATION_ONE = LOADED.get("HYDRATION")
LI_EXPULSION_ONE = LOADED.get("LI_EXPULSION_1")
LI_EXPULSION_TWO = LOADED.get("LI_EXPULSION_2")
SI_LITHIATION_ONE = LOADED.get("SI_LITHIATION")

