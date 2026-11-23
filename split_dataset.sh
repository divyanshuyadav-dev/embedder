
#!/usr/bin/env bash

set -euo pipefail

SOURCE="cattle_dataset"
DEST="cattle_dataset_split"

TRAIN_RATIO=0.8
SEED=42

# Check source
if [[ ! -d "$SOURCE" ]]; then
    echo "Error: $SOURCE not found!"
    exit 1
fi

if [[ -e "$DEST" ]]; then
    echo "Error: $DEST already exists."
    echo "Remove it manually if you want to regenerate the split."
    exit 1
fi

mkdir -p "$DEST/train" "$DEST/val"

export SEED TRAIN_RATIO

python - <<'PY'
import os
import random
import shutil
from pathlib import Path

source = Path("cattle_dataset")
destination = Path("cattle_dataset_split")

random.seed(int(os.environ["SEED"]))
ratio = float(os.environ["TRAIN_RATIO"])

extensions = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

total_images = 0
train_total = 0
val_total = 0
skipped = []

classes = sorted(
    p for p in source.iterdir()
    if p.is_dir()
)

for class_dir in classes:

    images = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    random.shuffle(images)

    if len(images) < 2:
        skipped.append(class_dir.name)
        continue

    split_index = round(len(images) * ratio)

    split_index = max(
        1,
        min(split_index, len(images) - 1)
    )

    train_images = images[:split_index]
    val_images = images[split_index:]

    for split_name, selected in [
        ("train", train_images),
        ("val", val_images)
    ]:

        output_dir = destination / split_name / class_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for image in selected:
            shutil.copy2(
                image,
                output_dir / image.name
            )

    train_total += len(train_images)
    val_total += len(val_images)
    total_images += len(images)

    print(
        f"{class_dir.name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}"
    )

print("\n========== SPLIT SUMMARY ==========")
print("Total identities:", len(classes))
print("Total images:", total_images)
print("Training images:", train_total)
print("Validation images:", val_total)
print("Skipped identities:", len(skipped))

if skipped:
    print("Skipped:", ", ".join(skipped))

print("\nDataset created at:", destination)
PY

echo "Done!"
