# resize_dataset.py
import os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm

SRC_DIR = "cattle_dataset_split"
DST_DIR = "cattle_dataset_128"
TARGET_SIZE = (128, 128)

def process_image(args):
    src_path, dst_path = args
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB").resize(TARGET_SIZE, Image.BILINEAR)
        img.save(dst_path, quality=95)

if __name__ == "__main__":
    tasks = []
    for root, _, files in os.walk(SRC_DIR):
        rel = os.path.relpath(root, SRC_DIR)
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                tasks.append((
                    os.path.join(root, f),
                    os.path.join(DST_DIR, rel, f)
                ))

    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_image, tasks), total=len(tasks)))
