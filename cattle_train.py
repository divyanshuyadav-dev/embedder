
"""
Cattle Face Recognition Training
MobileFaceNet + ArcFace + PyTorch

Dataset:
    cattle_dataset/
        train/
            cow_0001/*.jpg
            cow_0002/*.jpg
        val/
            cow_0001/*.jpg
            cow_0002/*.jpg

Pretrained model:
    mobilefacenet_scripted.pt

Important:
    Use identity-verified, near-duplicate-cleaned images.
    Split by recording session/video, not random frames.
"""

import os
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms

from PIL import Image
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "cattle_dataset_split"

PRETRAINED_MODEL = "mobilefacenet_scripted.pt"

OUTPUT_DIR = "cattle_checkpoints"

IMAGE_SIZE = 112

EPOCHS = 50

BATCH_SIZE = 32

# P identities x K images per identity
P = 8
K = 4

BACKBONE_LR = 0.00001
ARCFACE_LR = 0.001

WEIGHT_DECAY = 0.0001

ARCFACE_MARGIN = 0.5
ARCFACE_SCALE = 64.0

NUM_WORKERS = 4

FREEZE_BACKBONE_EPOCHS = 2

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# 3. DATASET
# ============================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "val"),
    transform=val_transform
)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError(
        "Training and validation class mappings differ. "
        "Ensure both splits contain the same identity folders."
    )

NUM_CLASSES = len(train_dataset.classes)

print("\nDataset information")
print("-------------------")
print("Identities:", NUM_CLASSES)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

if NUM_CLASSES < P:
    raise ValueError("Not enough identities for P-K sampling.")

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")


# ============================================================
# 4. P-K BATCH SAMPLER
# ============================================================

class PKBatchSampler(Sampler):

    def __init__(self, dataset, p=8, k=4):

        self.p = p
        self.k = k

        self.class_indices = {}

        for index, label in enumerate(dataset.targets):

            self.class_indices.setdefault(
                label, []
            ).append(index)

        self.classes = list(self.class_indices.keys())

        self.batch_size = p * k

        self.num_batches = max(
            1,
            len(dataset) // self.batch_size
        )

    def __iter__(self):

        for _ in range(self.num_batches):

            selected_classes = random.sample(
                self.classes, self.p
            )

            batch = []

            for class_id in selected_classes:

                indices = self.class_indices[class_id]

                # Sample without replacement whenever possible.
                if len(indices) >= self.k:
                    selected = random.sample(
                        indices, self.k
                    )
                else:
                    selected = random.choices(
                        indices, k=self.k
                    )

                batch.extend(selected)

            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


train_sampler = PKBatchSampler(
    train_dataset,
    p=P,
    k=K
)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# 5. LOAD PRETRAINED MOBILEFACENET
# ============================================================

if not os.path.exists(PRETRAINED_MODEL):
    raise FileNotFoundError(
        f"Pretrained model not found: {PRETRAINED_MODEL}"
    )

backbone = torch.jit.load(
    PRETRAINED_MODEL,
    map_location=device
)

backbone.to(device)

# Check that the checkpoint has trainable parameters.
parameters = list(backbone.parameters())

if not parameters:
    raise RuntimeError(
        "This TorchScript model exposes no parameters. "
        "Use a trainable PyTorch checkpoint instead."
    )

# Discover embedding dimension.
backbone.eval()

with torch.no_grad():

    dummy = torch.zeros(
        2, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=device
    )

    dummy_output = backbone(dummy)

    if isinstance(dummy_output, (tuple, list)):
        dummy_output = dummy_output[0]

    embedding_dim = dummy_output.flatten(1).shape[1]

print("\nMobileFaceNet loaded")
print("Embedding dimension:", embedding_dim)


# ============================================================
# 6. ARCFACE HEAD
# ============================================================

class ArcFace(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_classes,
        margin=0.5,
        scale=64.0
    ):

        super().__init__()

        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )

        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)

        self.mm = (
            math.sin(math.pi - margin) * margin
        )

    def forward(self, embeddings, labels=None):

        embeddings = F.normalize(
            embeddings, p=2, dim=1
        )

        weights = F.normalize(
            self.weight, p=2, dim=1
        )

        cosine = F.linear(embeddings, weights)

        cosine = cosine.clamp(
            -1.0 + 1e-7,
            1.0 - 1e-7
        )

        # Validation/inference: no angular margin.
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(
            1.0 - cosine.pow(2)
        )

        phi = (
            cosine * self.cos_m
            - sine * self.sin_m
        )

        phi = torch.where(
            cosine > self.th,
            phi,
            cosine - self.mm
        )

        one_hot = F.one_hot(
            labels,
            num_classes=self.weight.shape[0]
        ).to(cosine.dtype)

        logits = (
            one_hot * phi
            + (1.0 - one_hot) * cosine
        )

        return logits * self.scale


arcface = ArcFace(
    embedding_dim=embedding_dim,
    num_classes=NUM_CLASSES,
    margin=ARCFACE_MARGIN,
    scale=ARCFACE_SCALE
).to(device)


# ============================================================
# 7. OPTIMIZER AND LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone.parameters(),
            "lr": BACKBONE_LR
        },
        {
            "params": arcface.parameters(),
            "lr": ARCFACE_LR
        }
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)


# ============================================================
# 8. VALIDATION
# ============================================================

def validate():

    backbone.eval()
    arcface.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            embeddings = backbone(images)

            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[0]

            embeddings = embeddings.flatten(1)

            logits = arcface(
                embeddings,
                labels=None
            )

            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / total

    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 9. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(epoch, val_accuracy, filename):

    torch.save(
        {
            "epoch": epoch,

            "backbone": backbone.state_dict(),

            "arcface": arcface.state_dict(),

            "embedding_dim": embedding_dim,

            "num_classes": NUM_CLASSES,

            "class_to_idx": train_dataset.class_to_idx,

            "val_accuracy": val_accuracy
        },
        os.path.join(OUTPUT_DIR, filename)
    )


# ============================================================
# 10. TRAINING LOOP
# ============================================================

best_accuracy = 0.0

for epoch in range(1, EPOCHS + 1):

    # Initially train only the ArcFace head.
    # Then fine-tune the complete backbone.

    freeze_backbone = (
        epoch <= FREEZE_BACKBONE_EPOCHS
    )

    for parameter in backbone.parameters():
        parameter.requires_grad = not freeze_backbone

    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()

    arcface.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass through MobileFaceNet.
        embeddings = backbone(images)

        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[0]

        embeddings = embeddings.flatten(1)

        # ArcFace classification.
        logits = arcface(
            embeddings,
            labels
        )

        # Cross entropy.
        loss = criterion(logits, labels)

        # Backpropagation.
        loss.backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            [
                p for p in list(backbone.parameters())
                + list(arcface.parameters())
                if p.requires_grad
            ],
            max_norm=5.0
        )

        optimizer.step()

        running_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100.0 * correct / total

    # Validation.
    val_loss, val_accuracy = validate()

    scheduler.step(val_accuracy)

    print("\n" + "=" * 55)

    print(f"Epoch: {epoch}/{EPOCHS}")

    print(f"Train Loss: {train_loss:.4f}")

    print(f"Train Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")

    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    print(
        "Backbone LR:",
        optimizer.param_groups[0]["lr"]
    )

    print(
        "ArcFace LR:",
        optimizer.param_groups[1]["lr"]
    )

    print("=" * 55)

    # Save latest checkpoint.
    save_checkpoint(
        epoch,
        val_accuracy,
        "latest.pth"
    )

    # Save best checkpoint.
    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        save_checkpoint(
            epoch,
            val_accuracy,
            "best.pth"
        )

        print("New best model saved!")

print("\nTraining completed.")

print("Best validation accuracy:", best_accuracy)

print(
    "Best checkpoint:",
    os.path.join(OUTPUT_DIR, "best.pth")
)
