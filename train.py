import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from model import DualBranchCoAtNetPVTv2Classifier

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderWithTransform(Dataset):
    def __init__(self, root, transform=None):
        self.base_dataset = datasets.ImageFolder(root=root)
        self.transform = transform
        self.loader = self.base_dataset.loader
        self.samples = self.base_dataset.samples
        self.targets = self.base_dataset.targets
        self.classes = self.base_dataset.classes
        self.class_to_idx = self.base_dataset.class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_backbones_trainable(model, trainable: bool):
    for p in model.gelu_branch.cnn.parameters():
        p.requires_grad = trainable
    for p in model.gelu_branch.pvt_stage.parameters():
        p.requires_grad = trainable
    for p in model.elu_branch.cnn.parameters():
        p.requires_grad = trainable
    for p in model.elu_branch.pvt_stage.parameters():
        p.requires_grad = trainable


def build_optimizer(model, lr: float):
    return optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_name", type=str, default="best_model.pth")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--elu_alpha", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    full_dataset_train = ImageFolderWithTransform(args.data_dir, transform=train_transform)
    full_dataset_eval = ImageFolderWithTransform(args.data_dir, transform=eval_transform)
    targets = np.array(full_dataset_train.targets)
    all_indices = np.arange(len(targets))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(all_indices, targets))

    train_loader = DataLoader(
        Subset(full_dataset_train, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(full_dataset_eval, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Classes:", full_dataset_train.classes)
    print("Train size:", len(train_idx))
    print("Val size:", len(val_idx))

    model = DualBranchCoAtNetPVTv2Classifier(
        dropout=args.dropout,
        elu_alpha=args.elu_alpha,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    set_backbones_trainable(model, False)
    optimizer = build_optimizer(model, args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / args.save_name

    best_val_auc = -1.0

    for epoch in range(args.epochs):
        start_time = time.time()

        if epoch == args.warmup_epochs:
            print("Unfreezing both CoAtNet + PVTv2 branches")
            set_backbones_trainable(model, True)
            optimizer = build_optimizer(model, args.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=2, factor=0.5
            )

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

        scheduler.step(val_metrics["roc_auc"])
        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch+1}/{args.epochs}] | "
            f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | Train AUC: {train_metrics['roc_auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    print(f"\nBest validation AUC: {best_val_auc:.4f}")
    print(f"Best model saved at: {save_path}")


if __name__ == "__main__":
    main()
