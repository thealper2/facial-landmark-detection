"""
Keypoint Detection with Transfer Learning - Facial Landmark Detection
======================================================================
This module implements a complete pipeline for facial keypoint detection using
transfer learning with PyTorch. It leverages a pretrained ResNet backbone and
fine-tunes it to predict 68 facial landmarks from the Facial Key Point Detection dataset.

Dataset Structure:
    - images/         : Directory containing PNG face images (e.g., 00000.png)
    - all_data.json   : JSON file mapping image filenames to 68 (x, y) landmark pairs

Usage:
    python keypoint_detection.py --data_dir /path/to/dataset --epochs 30 --batch_size 32
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, UnidentifiedImageError
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("keypoint_detection.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_KEYPOINTS: int = 68  # 68 facial landmarks per image
OUTPUT_SIZE: int = NUM_KEYPOINTS * 2  # Flattened (x, y) pairs → 136 values
IMAGE_SIZE: int = 224  # ResNet input size
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# Facial landmark group indices for color-coded visualization
LANDMARK_GROUPS: dict[str, tuple[list[int], str]] = {
    "Jaw": (list(range(0, 17)), "#FF6B6B"),
    "Right Brow": (list(range(17, 22)), "#FFD93D"),
    "Left Brow": (list(range(22, 27)), "#FFD93D"),
    "Nose Bridge": (list(range(27, 31)), "#6BCB77"),
    "Nose Tip": (list(range(31, 36)), "#4D96FF"),
    "Right Eye": (list(range(36, 42)), "#FF922B"),
    "Left Eye": (list(range(42, 48)), "#CC5DE8"),
    "Outer Lips": (list(range(48, 60)), "#F06595"),
    "Inner Lips": (list(range(60, 68)), "#74C0FC"),
}


# ──────────────────────────────────────────────────────────────────────────────
# Data Validation
# ──────────────────────────────────────────────────────────────────────────────


def validate_dataset_paths(data_dir: str) -> tuple[Path, Path]:
    """
    Validate that the dataset directory contains the required structure.

    Expected layout:
        <data_dir>/images/   - directory of PNG face images
        <data_dir>/all_data.json - landmark annotations

    Args:
        data_dir: Root directory of the dataset.

    Returns:
        Tuple of (images_dir, json_path) as resolved Path objects.

    Raises:
        FileNotFoundError: If required paths are missing.
        ValueError:        If the directory structure is invalid.
    """
    root = Path(data_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    images_dir = root / "images"
    json_path = root / "all_data.json"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not json_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {json_path}")

    png_files = list(images_dir.glob("*.png"))
    if not png_files:
        raise ValueError(f"No PNG images found in: {images_dir}")

    logger.info("Dataset validated — %d images found.", len(png_files))
    return images_dir, json_path


def validate_annotation(
    key: str,
    entry: Any,
    images_dir: Path,
) -> bool:
    """
    Validate a single annotation entry from all_data.json.

    Args:
        key:        The string key (e.g., "0", "1", ...) for logging.
        entry:      The annotation dict with 'file_name' and 'face_landmarks'.
        images_dir: Path to the images directory.

    Returns:
        True if the entry is valid, False otherwise (with a warning logged).
    """
    if not isinstance(entry, dict):
        logger.warning("Skipping key %s: entry is not a dict.", key)
        return False

    file_name = entry.get("file_name")
    landmarks = entry.get("face_landmarks")

    if not isinstance(file_name, str) or not file_name.strip():
        logger.warning("Skipping key %s: missing or empty 'file_name'.", key)
        return False

    if not (images_dir / file_name).exists():
        logger.warning("Skipping key %s: image '%s' not found.", key, file_name)
        return False

    if not isinstance(landmarks, list) or len(landmarks) != NUM_KEYPOINTS:
        logger.warning(
            "Skipping key %s: expected %d landmarks, got %s.",
            key,
            NUM_KEYPOINTS,
            len(landmarks) if isinstance(landmarks, list) else type(landmarks),
        )
        return False

    for idx, pt in enumerate(landmarks):
        if not (isinstance(pt, list) and len(pt) == 2):
            logger.warning("Skipping key %s: landmark %d has invalid format.", key, idx)
            return False
        if not all(isinstance(c, (int, float)) for c in pt):
            logger.warning(
                "Skipping key %s: landmark %d has non-numeric values.", key, idx
            )
            return False

    return True


def load_annotations(json_path: Path, images_dir: Path) -> list[dict[str, Any]]:
    """
    Load and validate the full annotation JSON file.

    The JSON is expected to have structure:
        { "root": { "0": { "file_name": ..., "face_landmarks": [...] }, ... } }

    Args:
        json_path:  Path to all_data.json.
        images_dir: Path to the images directory for existence checks.

    Returns:
        List of validated annotation dicts, each with 'file_name' and 'face_landmarks'.

    Raises:
        ValueError: If the JSON structure is unrecognised or no valid entries remain.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw: Any = json.load(f)

    # Unwrap optional "root" wrapper
    if isinstance(raw, dict) and "root" in raw and isinstance(raw["root"], dict):
        data_dict: dict[str, Any] = raw["root"]
    elif isinstance(raw, dict):
        data_dict = raw
    else:
        raise ValueError(
            "Unexpected JSON structure: top-level must be a dict (with optional 'root' key)."
        )

    valid_entries: list[dict[str, Any]] = []
    for key, entry in data_dict.items():
        if validate_annotation(key, entry, images_dir):
            valid_entries.append(entry)

    if not valid_entries:
        raise ValueError("No valid annotation entries found after validation.")

    logger.info("Loaded %d valid annotations.", len(valid_entries))
    return valid_entries


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


def build_transforms(augment: bool = False) -> transforms.Compose:
    """
    Build the image transform pipeline.

    For training, optional augmentation (horizontal flip, color jitter, rotation)
    is applied before normalization. Validation/test transforms skip augmentation.

    Args:
        augment: Whether to include data augmentation transforms.

    Returns:
        A composed torchvision transform.
    """
    base = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(
                p=0.0
            ),  # Flipping would mirror landmarks; skip
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.05),
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)


class FacialKeypointDataset(Dataset):
    """
    PyTorch Dataset for Facial Keypoint Detection.

    Each item returns:
        image   : Float tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized.
        keypoints: Float tensor of shape (136,), landmark (x, y) pairs normalised
                   to [-1, 1] by dividing by (original_width / 2) or (original_height / 2)
                   and subtracting 1, so the model outputs are scale-invariant.
    """

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        images_dir: Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Args:
            annotations: Validated annotation dicts.
            images_dir:  Directory where PNG images live.
            transform:   Optional image transform pipeline.
        """
        self.annotations = annotations
        self.images_dir = images_dir
        self.transform = transform or build_transforms(augment=False)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one sample: open image, normalise landmarks, apply transforms.

        Landmark normalisation:  coord_norm = (coord / (dim / 2)) - 1
        This maps raw pixel coords into [-1, 1] regardless of original image size.
        """
        entry = self.annotations[idx]
        img_path = self.images_dir / entry["file_name"]

        # ── Load image ────────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            logger.error("Cannot open image %s: %s", img_path, exc)
            # Return a zero-filled fallback to avoid crashing the DataLoader
            image = Image.fromarray(
                np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            )

        orig_w, orig_h = image.size  # Original dimensions before resize

        # ── Process landmarks ─────────────────────────────────────────────────
        landmarks: list[list[float]] = entry["face_landmarks"]
        coords: list[float] = []
        for x, y in landmarks:
            # Normalise each coordinate to [-1, 1]
            coords.append(float(x) / (orig_w / 2.0) - 1.0)
            coords.append(float(y) / (orig_h / 2.0) - 1.0)

        keypoints = torch.tensor(coords, dtype=torch.float32)

        # ── Apply image transforms ────────────────────────────────────────────
        if self.transform:
            image = self.transform(image)

        return image, keypoints


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────


def build_model(
    num_outputs: int = OUTPUT_SIZE,
    freeze_backbone: bool = True,
    dropout_p: float = 0.4,
) -> nn.Module:
    """
    Build a ResNet-50 transfer learning model for keypoint regression.

    Architecture:
        - ResNet-50 backbone pretrained on ImageNet (frozen by default).
        - Custom regression head: Linear → BN → ReLU → Dropout → Linear → Tanh.
        - Tanh activation constrains outputs to [-1, 1], matching normalised landmarks.

    Args:
        num_outputs:      Number of output values (default 136 = 68 keypoints × 2).
        freeze_backbone:  If True, freeze all backbone layers initially.
        dropout_p:        Dropout probability in the regression head.

    Returns:
        nn.Module ready for training.
    """
    backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        # Unfreeze the last residual block (layer4) for fine-tuning
        for param in backbone.layer4.parameters():
            param.requires_grad = True

    in_features: int = backbone.fc.in_features

    # Replace the classification head with a regression head
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_p / 2),
        nn.Linear(256, num_outputs),
        nn.Tanh(),  # Constrains predictions to [-1, 1]
    )

    return backbone


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all backbone parameters for full fine-tuning.

    Called after initial training with a frozen backbone so the model can
    perform a second fine-tuning pass with a smaller learning rate.

    Args:
        model: The ResNet model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All backbone parameters unfrozen for fine-tuning.")


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────


def compute_nme(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_size: int = IMAGE_SIZE,
) -> float:
    """
    Normalised Mean Error (NME) — primary landmark detection metric.

    NME is the mean Euclidean distance between predicted and ground-truth
    landmarks, normalised by the inter-ocular distance (distance between the
    outer eye corners: landmark 36 and landmark 45).

    A lower NME indicates better accuracy; values < 0.06 are considered good.

    Args:
        preds:      Predicted keypoints, shape (B, 136), in [-1, 1] space.
        targets:    Ground-truth keypoints, shape (B, 136), in [-1, 1] space.
        image_size: Pixel size used to convert normalised coords back to pixels.

    Returns:
        Mean NME across the batch (float).
    """
    B = preds.size(0)

    # Denormalise to pixel space: coord_px = (coord_norm + 1) * (image_size / 2)
    scale = image_size / 2.0
    p = (preds.detach().cpu().float() + 1.0) * scale  # (B, 136)
    t = (targets.detach().cpu().float() + 1.0) * scale

    # Reshape to (B, 68, 2)
    p = p.view(B, NUM_KEYPOINTS, 2)
    t = t.view(B, NUM_KEYPOINTS, 2)

    # Per-sample inter-ocular distance: landmarks 36 (right outer) and 45 (left outer)
    iod = torch.norm(t[:, 45, :] - t[:, 36, :], dim=1)  # (B,)
    iod = iod.clamp(min=1e-6)  # Prevent division by zero

    # Mean per-landmark Euclidean distance for each sample
    dist = torch.norm(p - t, dim=2).mean(dim=1)  # (B,)

    nme = (dist / iod).mean().item()
    return nme


def compute_auc(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_size: int = IMAGE_SIZE,
    threshold_range: tuple[float, float] = (0.0, 0.1),
    num_thresholds: int = 100,
) -> float:
    """
    Area Under the Curve (AUC) for the Cumulative Error Distribution (CED).

    AUC@0.1 measures the proportion of test images whose NME is below a
    threshold, integrated over the threshold range [0, 0.1]. Higher is better.

    Args:
        preds:           Predicted keypoints (B, 136).
        targets:         Ground-truth keypoints (B, 136).
        image_size:      Pixel size for denormalisation.
        threshold_range: (min_thresh, max_thresh) for the CED curve.
        num_thresholds:  Number of threshold steps.

    Returns:
        AUC value (float in [0, 1]).
    """
    B = preds.size(0)
    scale = image_size / 2.0

    p = ((preds.detach().cpu().float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)
    t = ((targets.detach().cpu().float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)

    iod = torch.norm(t[:, 45, :] - t[:, 36, :], dim=1).clamp(min=1e-6)
    per_sample_nme = torch.norm(p - t, dim=2).mean(dim=1) / iod  # (B,)
    per_sample_nme = per_sample_nme.numpy()

    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)
    ced = np.array([(per_sample_nme <= thr).mean() for thr in thresholds])

    # Normalise AUC to [0, 1] by dividing by number of threshold steps
    auc = float(np.trapz(ced, thresholds) / (threshold_range[1] - threshold_range[0]))
    return auc


def compute_per_landmark_error(
    preds: torch.Tensor,
    targets: torch.Tensor,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Compute the mean pixel error for each of the 68 landmarks independently.

    Useful for diagnosing which facial regions are hardest to predict.

    Args:
        preds:      Predicted keypoints (B, 136).
        targets:    Ground-truth keypoints (B, 136).
        image_size: Pixel size for denormalisation.

    Returns:
        NumPy array of shape (68,) with mean pixel error per landmark.
    """
    B = preds.size(0)
    scale = image_size / 2.0

    p = ((preds.detach().cpu().float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)
    t = ((targets.detach().cpu().float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)

    # Mean Euclidean error across the batch for each landmark
    per_lm_error = torch.norm(p - t, dim=2).mean(dim=0).numpy()  # (68,)
    return per_lm_error


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Run one training epoch over the given DataLoader.

    Args:
        model:     The keypoint model.
        loader:    Training DataLoader.
        criterion: Loss function (e.g., MSELoss or SmoothL1Loss).
        optimizer: Parameter optimiser.
        device:    CUDA or CPU device.

    Returns:
        Tuple of (mean_loss, mean_nme, mean_auc) for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_nme = 0.0
    total_auc = 0.0
    n_batches = 0

    for images, keypoints in loader:
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
        preds = model(images)
        loss = criterion(preds, keypoints)
        loss.backward()

        # Gradient clipping prevents exploding gradients during fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_nme += compute_nme(preds, keypoints)
        total_auc += compute_auc(preds, keypoints)
        n_batches += 1

    return total_loss / n_batches, total_nme / n_batches, total_auc / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate the model on a validation or test DataLoader.

    Args:
        model:     The keypoint model (eval mode is set internally).
        loader:    Validation or test DataLoader.
        criterion: Loss function.
        device:    CUDA or CPU device.

    Returns:
        Tuple of (mean_loss, mean_nme, mean_auc) for the split.
    """
    model.eval()
    total_loss = 0.0
    total_nme = 0.0
    total_auc = 0.0
    n_batches = 0

    for images, keypoints in loader:
        images = images.to(device, non_blocking=True)
        keypoints = keypoints.to(device, non_blocking=True)

        preds = model(images)
        loss = criterion(preds, keypoints)

        total_loss += loss.item()
        total_nme += compute_nme(preds, keypoints)
        total_auc += compute_auc(preds, keypoints)
        n_batches += 1

    return total_loss / n_batches, total_nme / n_batches, total_auc / n_batches


# ──────────────────────────────────────────────────────────────────────────────
# Full Training Loop
# ──────────────────────────────────────────────────────────────────────────────


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    fine_tune_epoch: int = 15,
    fine_tune_lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
) -> dict[str, list[float]]:
    """
    Full two-phase training loop.

    Phase 1 (epochs 1 → fine_tune_epoch):  Frozen backbone, train head only.
    Phase 2 (fine_tune_epoch → epochs):    Unfreeze backbone, full fine-tuning.

    Args:
        model:            The keypoint model.
        train_loader:     DataLoader for training split.
        val_loader:       DataLoader for validation split.
        device:           Compute device.
        epochs:           Total number of training epochs.
        lr:               Initial learning rate for Phase 1.
        fine_tune_epoch:  Epoch at which Phase 2 begins.
        fine_tune_lr:     Learning rate for Phase 2 fine-tuning.
        checkpoint_dir:   Directory to save best model checkpoints.

    Returns:
        History dict with keys: train_loss, val_loss, train_nme, val_nme,
        train_auc, val_auc, lr.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    criterion = nn.SmoothL1Loss()  # More robust to outliers than MSE
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_nme": [],
        "val_nme": [],
        "train_auc": [],
        "val_auc": [],
        "lr": [],
    }
    best_val_nme: float = float("inf")
    best_ckpt_path = Path(checkpoint_dir) / "best_model.pth"

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── Phase transition: unfreeze backbone ───────────────────────────────
        if epoch == fine_tune_epoch:
            logger.info("=== Phase 2: Unfreezing backbone for fine-tuning ===")
            unfreeze_backbone(model)
            # Re-create optimiser with a smaller learning rate for all parameters
            optimizer = optim.AdamW(
                model.parameters(), lr=fine_tune_lr, weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=epochs - fine_tune_epoch + 1, eta_min=1e-6
            )

        # ── Train & Validate ──────────────────────────────────────────────────
        tr_loss, tr_nme, tr_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        vl_loss, vl_nme, vl_auc = evaluate(model, val_loader, criterion, device)

        # Scheduler step (ReduceLROnPlateau needs val_loss; CosineAnnealingLR does not)
        if epoch < fine_tune_epoch:
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # ── Logging ───────────────────────────────────────────────────────────
        logger.info(
            "Epoch %03d/%03d | "
            "Loss: %.4f / %.4f | "
            "NME: %.4f / %.4f | "
            "AUC: %.4f / %.4f | "
            "LR: %.2e | %.1fs",
            epoch,
            epochs,
            tr_loss,
            vl_loss,
            tr_nme,
            vl_nme,
            tr_auc,
            vl_auc,
            current_lr,
            elapsed,
        )

        # ── History tracking ──────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_nme"].append(tr_nme)
        history["val_nme"].append(vl_nme)
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(vl_auc)
        history["lr"].append(current_lr)

        # ── Checkpoint best model ─────────────────────────────────────────────
        if vl_nme < best_val_nme:
            best_val_nme = vl_nme
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_nme": vl_nme,
                    "val_loss": vl_loss,
                },
                best_ckpt_path,
            )
            logger.info("  ✓ Best model saved (val_nme=%.4f)", best_val_nme)

    logger.info("Training complete. Best val NME: %.4f", best_val_nme)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────


def _denormalise_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and convert a CHW float tensor to HWC uint8.

    Args:
        tensor: Float tensor of shape (3, H, W).

    Returns:
        NumPy array of shape (H, W, 3) with dtype uint8, values in [0, 255].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu().float() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _landmarks_to_pixels(
    coords: torch.Tensor,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert normalised [-1, 1] landmark coordinates to pixel coordinates.

    Args:
        coords:     Flat tensor of 136 values (68 x,y pairs) in [-1, 1].
        image_size: Side length of the square image in pixels.

    Returns:
        NumPy array of shape (68, 2) in pixel space.
    """
    pts = (coords.cpu().float().numpy() + 1.0) * (image_size / 2.0)
    return pts.reshape(NUM_KEYPOINTS, 2)


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str = "training_history.png",
) -> None:
    """
    Plot and save training/validation loss, NME, AUC, and learning rate curves.

    Args:
        history:   Dict returned by train().
        save_path: File path for the saved figure.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Training History — Facial Keypoint Detection", fontsize=16, fontweight="bold"
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train", color="#4D96FF", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Val", color="#FF6B6B", linewidth=2)
    ax.set_title("SmoothL1 Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── NME ───────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(epochs, history["train_nme"], label="Train", color="#4D96FF", linewidth=2)
    ax.plot(epochs, history["val_nme"], label="Val", color="#FF6B6B", linewidth=2)
    ax.axhline(
        y=0.06, color="green", linestyle="--", alpha=0.7, label="Good threshold (0.06)"
    )
    ax.set_title("Normalised Mean Error (NME ↓)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NME")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── AUC ───────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(epochs, history["train_auc"], label="Train", color="#4D96FF", linewidth=2)
    ax.plot(epochs, history["val_auc"], label="Val", color="#FF6B6B", linewidth=2)
    ax.set_title("AUC@0.1 (↑ is better)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Learning Rate ─────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.semilogy(epochs, history["lr"], color="#6BCB77", linewidth=2)
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR (log scale)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training history saved → %s", save_path)


def visualise_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 8,
    save_path: str = "predictions.png",
) -> None:
    """
    Visualise predicted vs ground-truth landmarks on a grid of sample images.

    Predictions are shown in colour-coded groups; ground truth in white.

    Args:
        model:     Trained keypoint model.
        loader:    DataLoader (validation or test) to sample from.
        device:    Compute device.
        n_samples: Number of images to plot (must be ≤ batch size).
        save_path: File path for the saved figure.
    """
    model.eval()
    images_batch, kp_batch = next(iter(loader))
    n = min(n_samples, len(images_batch))

    images_batch = images_batch[:n].to(device)
    kp_batch = kp_batch[:n]

    with torch.no_grad():
        preds_batch = model(images_batch).cpu()

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        img = _denormalise_image(images_batch[i].cpu())
        pred_pts = _landmarks_to_pixels(preds_batch[i])
        gt_pts = _landmarks_to_pixels(kp_batch[i])

        ax.imshow(img)
        ax.axis("off")

        # Plot ground-truth landmarks (white circles)
        ax.scatter(gt_pts[:, 0], gt_pts[:, 1], s=8, c="white", zorder=3, alpha=0.8)

        # Plot predicted landmarks in colour-coded groups
        for group_name, (indices, color) in LANDMARK_GROUPS.items():
            ax.scatter(
                pred_pts[indices, 0],
                pred_pts[indices, 1],
                s=10,
                c=color,
                zorder=4,
                alpha=0.9,
            )

        ax.set_title(f"Sample {i + 1}", fontsize=9)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # Legend for landmark groups
    patches = [
        mpatches.Patch(color=color, label=name)
        for name, (_, color) in LANDMARK_GROUPS.items()
    ]
    patches.append(mpatches.Patch(color="white", label="Ground Truth"))
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.suptitle("Predicted (coloured) vs Ground Truth (white) Landmarks", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Prediction visualisation saved → %s", save_path)


def visualise_per_landmark_error(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "per_landmark_error.png",
) -> None:
    """
    Bar chart of mean pixel error for each of the 68 landmarks.

    Bars are colour-coded by facial region to make it easy to see which
    regions are predicted with lower accuracy.

    Args:
        model:     Trained keypoint model.
        loader:    DataLoader to evaluate on.
        device:    Compute device.
        save_path: File path for the saved figure.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, keypoints in loader:
            preds = model(images.to(device)).cpu()
            all_preds.append(preds)
            all_targets.append(keypoints)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    per_lm_error = compute_per_landmark_error(preds_cat, targets_cat)

    # Assign a colour to each landmark based on its group
    landmark_colors = ["#AAAAAA"] * NUM_KEYPOINTS
    for _, (indices, color) in LANDMARK_GROUPS.items():
        for idx in indices:
            landmark_colors[idx] = color

    fig, ax = plt.subplots(figsize=(18, 5))
    bars = ax.bar(
        range(NUM_KEYPOINTS), per_lm_error, color=landmark_colors, edgecolor="none"
    )
    ax.set_xlabel("Landmark Index", fontsize=11)
    ax.set_ylabel("Mean Pixel Error", fontsize=11)
    ax.set_title("Per-Landmark Mean Pixel Error", fontsize=13, fontweight="bold")
    ax.axhline(
        per_lm_error.mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean ({per_lm_error.mean():.2f} px)",
    )
    ax.set_xticks(range(0, NUM_KEYPOINTS, 5))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    patches = [
        mpatches.Patch(color=color, label=name)
        for name, (_, color) in LANDMARK_GROUPS.items()
    ]
    ax.legend(
        handles=patches + ax.get_legend_handles_labels()[0][-1:],
        fontsize=8,
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Per-landmark error chart saved → %s", save_path)


def plot_ced_curve(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "ced_curve.png",
) -> None:
    """
    Plot the Cumulative Error Distribution (CED) curve.

    The CED shows the fraction of test images whose NME is below each threshold.
    A model that shifts the curve up-and-left has fewer failures.

    Args:
        model:     Trained keypoint model.
        loader:    DataLoader to evaluate on.
        device:    Compute device.
        save_path: File path for the saved figure.
    """
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, keypoints in loader:
            preds = model(images.to(device)).cpu()
            all_preds.append(preds)
            all_targets.append(keypoints)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    B = preds_cat.size(0)
    scale = IMAGE_SIZE / 2.0
    p = ((preds_cat.float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)
    t = ((targets_cat.float() + 1.0) * scale).view(B, NUM_KEYPOINTS, 2)
    iod = torch.norm(t[:, 45, :] - t[:, 36, :], dim=1).clamp(min=1e-6)
    per_sample_nme = (torch.norm(p - t, dim=2).mean(dim=1) / iod).numpy()

    thresholds = np.linspace(0.0, 0.10, 200)
    ced = [(per_sample_nme <= thr).mean() for thr in thresholds]

    auc = float(np.trapz(ced, thresholds) / 0.10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        thresholds, ced, color="#4D96FF", linewidth=2.5, label=f"Model (AUC={auc:.3f})"
    )
    ax.fill_between(thresholds, ced, alpha=0.1, color="#4D96FF")
    ax.axvline(
        0.06, color="green", linestyle="--", alpha=0.7, label="Good threshold (0.06)"
    )
    ax.axvline(
        0.10,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label="Acceptable threshold (0.10)",
    )
    ax.set_xlabel("NME Threshold", fontsize=12)
    ax.set_ylabel("Fraction of Samples", fontsize=12)
    ax.set_title("Cumulative Error Distribution (CED)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 0.10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("CED curve saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Facial Keypoint Detection via Transfer Learning (ResNet-50)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root dataset directory containing images/ and all_data.json.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--fine_tune_lr", type=float, default=1e-4, help="Fine-tuning learning rate."
    )
    parser.add_argument(
        "--fine_tune_epoch",
        type=int,
        default=15,
        help="Epoch at which to unfreeze the backbone for full fine-tuning.",
    )
    parser.add_argument(
        "--val_split", type=float, default=0.15, help="Fraction for validation."
    )
    parser.add_argument(
        "--test_split", type=float, default=0.10, help="Fraction for testing."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="DataLoader worker count."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint save dir."
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Plot save dir."
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip training; load checkpoint and run evaluation/visualisation only.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point: validates data, builds loaders, trains the model,
    evaluates on the test set, and saves all plots.
    """
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info(
            "  GPU: %s | Memory: %.1f GB",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    images_dir, json_path = validate_dataset_paths(args.data_dir)
    annotations = load_annotations(json_path, images_dir)

    n = len(annotations)
    n_val = int(n * args.val_split)
    n_test = int(n * args.test_split)
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"Not enough data: {n} samples, splits require {n_val + n_test} for val+test."
        )
    logger.info("Split: %d train | %d val | %d test", n_train, n_val, n_test)

    full_dataset = FacialKeypointDataset(annotations, images_dir, transform=None)
    train_ds, val_ds, test_ds = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Override transforms: augmentation only for training
    train_ds.dataset.transform = build_transforms(augment=True)

    loader_kwargs: dict[str, Any] = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(freeze_backbone=not args.skip_train).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: ResNet-50 | Params: %s total | %s trainable",
        f"{total_params:,}",
        f"{trainable_params:,}",
    )

    # ── Train or Load ─────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint_dir) / "best_model.pth"

    if args.skip_train:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"--skip_train set but no checkpoint found at {ckpt_path}"
            )
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded checkpoint from epoch %d (val_nme=%.4f).",
            ckpt["epoch"],
            ckpt["val_nme"],
        )
        history: dict[str, list[float]] = {}
    else:
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            fine_tune_epoch=args.fine_tune_epoch,
            fine_tune_lr=args.fine_tune_lr,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Load the best checkpoint for final evaluation
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Best checkpoint loaded for final evaluation.")

    # ── Final Test Evaluation ─────────────────────────────────────────────────
    criterion = nn.SmoothL1Loss()
    test_loss, test_nme, test_auc = evaluate(model, test_loader, criterion, device)
    logger.info(
        "=== TEST SET RESULTS ===\n"
        "  SmoothL1 Loss : %.4f\n"
        "  NME           : %.4f\n"
        "  AUC@0.1       : %.4f",
        test_loss,
        test_nme,
        test_auc,
    )

    # ── Visualisations ────────────────────────────────────────────────────────
    if history:
        plot_training_history(
            history,
            save_path=os.path.join(args.output_dir, "training_history.png"),
        )

    visualise_predictions(
        model,
        test_loader,
        device,
        n_samples=8,
        save_path=os.path.join(args.output_dir, "predictions.png"),
    )
    visualise_per_landmark_error(
        model,
        test_loader,
        device,
        save_path=os.path.join(args.output_dir, "per_landmark_error.png"),
    )
    plot_ced_curve(
        model,
        test_loader,
        device,
        save_path=os.path.join(args.output_dir, "ced_curve.png"),
    )

    logger.info("All outputs saved to: %s/", args.output_dir)


if __name__ == "__main__":
    main()
