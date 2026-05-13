import os
from PIL import Image
import numpy as np


def normalize_stack(stack):
    stack = stack.astype(np.float32)
    if stack.max() > 1.5:
        stack = stack / stack.max()
    else:
        stack = np.clip(stack, 0, 1)

    return stack


def normalize_all_videos(trainingSet):
    normed = {}
    for t, stack in trainingSet.items():
        normed[t] = normalize_stack(stack)

    return normed


class SEMNoiseDataset:

    def __init__(self,
                 folder_path):

        self.folder_path = folder_path
        stacks_raw = self.load_all_tiffs()
        self.stacks = normalize_all_videos(stacks_raw)

    def load_all_tiffs(self):
        stacks = {}
        for filename in os.listdir(self.folder_path):

            if not filename.lower().endswith((".tif", ".tiff")):
                continue

            path = os.path.join(self.folder_path,filename)
            dwell = int(os.path.splitext(filename)[0])

            with Image.open(path) as img:
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)

                    frames.append(
                        np.array(img)
                    )
            stacks[dwell] = np.stack(frames)

        return stacks