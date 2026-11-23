"""
Cattle Face Recognition Training
MobileFaceNet (Native PyTorch) + ArcFace + Open-Set Validation

Dataset structure (split by UNSEEN identities for open-set validation):
    cattle_dataset_split/
        train/
            cow_0001/*.jpg
            cow_0002/*.jpg
        val/
            cow_0101/*.jpg   # Unseen cows not present in train/
            cow_0102/*.jpg
"""

import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_ROOT = "cattle_dataset_split"
PRETRAINED_WEIGHTS = "mobilefacenet_weights.pth"  # Set to None to train from scratch
OUTPUT_DIR = "cattle_checkpoints"

IMAGE_SIZE = 112
EMBEDDING_DIM = 512

EPOCHS = 50
BATCH_SIZE = 32
FREEZE_BACKBONE_EPOCHS = 2

BACKBONE_LR = 1e-4
ARCFACE_LR = 1e-3
WEIGHT_DECAY = 1e-4

# Safer defaults for livestock datasets; increase to 0.5 / 64.0 if you have >500 identities
ARCFACE_MARGIN = 0.3
ARCFACE_SCALE = 32.0

NUM_WORKERS = 4
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# 2. IMAGE TRANSFORMS (NO HORIZONTAL FLIP)
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.02
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# ============================================================
# 3. DATASET AND DATALOADERS
# ============================================================

train_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=os.path.join(DATA_ROOT, "val"),
    transform=val_transform
)

if len(train_dataset) == 0 or len(val_dataset) == 0:
    raise ValueError("Training or validation dataset is empty.")

NUM_CLASSES = len(train_dataset.classes)

print("\nDataset information")
print("-------------------")
print("Train identities:", NUM_CLASSES)
print("Train images:", len(train_dataset))
print("Val identities:", len(val_dataset.classes))
print("Val images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
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
# 4. NATIVE PYTORCH MOBILEFACENET
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthWise(nn.Module):
    def __init__(self, in_c, out_c, residual=False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super().__init__()
        self.residual = residual
        self.conv = ConvBlock(in_c, groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = ConvBlock(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = LinearBlock(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        return x + out if self.residual else out


class Residual(nn.Module):
    def __init__(self, channels, num_blocks, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.model = nn.Sequential(*[
            DepthWise(channels, channels, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.model(x)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = ConvBlock(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = DepthWise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_blocks=4, groups=128)
        self.conv_34 = DepthWise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_blocks=6, groups=256)
        self.conv_45 = DepthWise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_blocks=2, groups=256)
        self.conv_6_sep = ConvBlock(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = LinearBlock(512, 512, groups=512, kernel=(7, 7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = nn.Flatten()
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


backbone = MobileFaceNet(embedding_size=EMBEDDING_DIM).to(device)

if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
    ckpt = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    state_dict = ckpt["backbone"] if isinstance(ckpt, dict) and "backbone" in ckpt else ckpt
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    print(f"\nLoaded pretrained weights from {PRETRAINED_WEIGHTS}")
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
else:
    print("\nNo pretrained .pth weights found; training MobileFaceNet from scratch.")


# ============================================================
# 5. ARCFACE HEAD
# ============================================================

class ArcFace(nn.Module):
    def __init__(self, embedding_dim, num_classes, margin=0.3, scale=32.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(1.0 - cosine.pow(2))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(cosine.dtype)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


arcface = ArcFace(
    embedding_dim=EMBEDDING_DIM,
    num_classes=NUM_CLASSES,
    margin=ARCFACE_MARGIN,
    scale=ARCFACE_SCALE
).to(device)


# ============================================================
# 6. OPTIMIZER AND SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    [
        {"params": backbone.parameters(), "lr": BACKBONE_LR},
        {"params": arcface.parameters(), "lr": ARCFACE_LR}
    ],
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(1, EPOCHS - FREEZE_BACKBONE_EPOCHS),
    eta_min=1e-6
)


# ============================================================
# 7. OPEN-SET VALIDATION (ROC-AUC & RANK-1 RETRIEVAL)
# ============================================================

@torch.no_grad()
def validate():
    backbone.eval()

    all_embeddings = []
    all_labels = []

    for images, labels in val_loader:
        images = images.to(device)
        embeddings = F.normalize(backbone(images), p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    labels = torch.cat(all_labels, dim=0)          # (N,)

    # Pairwise cosine similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T)

    # 1. Rank-1 Retrieval Accuracy (leave-one-out nearest neighbor)
    sim_no_diag = sim_matrix.clone()
    sim_no_diag.fill_diagonal_(-1e9)
    nn_indices = sim_no_diag.argmax(dim=1)
    rank1_acc = (labels[nn_indices] == labels).float().mean().item() * 100.0

    # 2. Pairwise Verification ROC-AUC
    n = labels.size(0)
    rows, cols = torch.triu_indices(n, n, offset=1)
    pair_scores = sim_matrix[rows, cols].numpy()
    pair_labels = (labels[rows] == labels[cols]).int().numpy()

    if len(set(pair_labels)) < 2:
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(pair_labels, pair_scores)

    return roc_auc, rank1_acc


# ============================================================
# 8. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(epoch, val_auc, rank1_acc, filename):
    torch.save(
        {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "arcface": arcface.state_dict(),
            "embedding_dim": EMBEDDING_DIM,
            "num_classes": NUM_CLASSES,
            "class_to_idx": train_dataset.class_to_idx,
            "val_auc": val_auc,
            "val_rank1": rank1_acc
        },
        os.path.join(OUTPUT_DIR, filename)
    )


# ============================================================
# 9. TRAINING LOOP
# ============================================================

best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    freeze_backbone = (epoch <= FREEZE_BACKBONE_EPOCHS)

    for param in backbone.parameters():
        param.requires_grad = not freeze_backbone

    if freeze_backbone:
        backbone.eval()
    else:
        backbone.train()

    arcface.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        embeddings = backbone(images)
        logits = arcface(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in list(backbone.parameters()) + list(arcface.parameters()) if p.requires_grad],
            max_norm=5.0
        )

        optimizer.step()

        running_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total

    val_auc, val_rank1 = validate()

    # Only step cosine schedule after warmup/frozen epochs
    if not freeze_backbone:
        scheduler.step()

    print("\n" + "=" * 55)
    print(f"Epoch: {epoch}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val ROC-AUC: {val_auc:.4f} | Val Rank-1: {val_rank1:.2f}%")
    print(f"Backbone LR: {optimizer.param_groups[0]['lr']:.6f} | ArcFace LR: {optimizer.param_groups[1]['lr']:.6f}")
    print("=" * 55)

    save_checkpoint(epoch, val_auc, val_rank1, "latest.pth")

    if val_auc > best_auc:
        best_auc = val_auc
        save_checkpoint(epoch, val_auc, val_rank1, "best.pth")
        print("New best model saved!")

print("\nTraining completed.")
print(f"Best Validation ROC-AUC: {best_auc:.4f}")
print("Best checkpoint:", os.path.join(OUTPUT_DIR, "best.pth"))
