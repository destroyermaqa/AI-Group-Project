
#initials

import subprocess, sys

def install(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

REQUIRED = {
    "torch"      : "torch",
    "torchvision": "torchvision",
    "numpy"      : "numpy",
    "matplotlib" : "matplotlib",
    "PIL"        : "Pillow",
}

def check_dependencies():
    print("=" * 60)
    print("Checking...")
    print("=" * 60)
    for import_name, pip_name in REQUIRED.items():
        try:
            __import__(import_name)
            print(f"{pip_name:15s} already installed")
        except ImportError:
            print(f"{pip_name:15s} installing...", end=" ", flush=True)
            install(pip_name)
            print("Ready")
    print()


#Importing dataset (We will use Pet dataset)

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


#Preparation

FAST_CONFIG = {
    "data_root"    : "./data",
    "img_size"     : 64,          
    "batch_size"   : 32,          
    "val_split"    : 0.2,
    "epochs"       : 5,           
    "lr"           : 1e-3,
    "download"     : True,
    "num_workers"  : 0,           
    "subset_size"  : 1500,        
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "save_path"    : "unet_pet_segmentation.pth",
}

FULL_CONFIG = {
    "data_root"    : "./data",
    "img_size"     : 128,
    "batch_size"   : 64,
    "val_split"    : 0.2,
    "epochs"       : 5,
    "lr"           : 1e-3,
    "download"     : True,
    "num_workers"  : 2,
    "subset_size"  : None,        
    "device"       : "cuda" if torch.cuda.is_available() else "cpu",
    "save_path"    : "unet_pet_segmentation.pth",
}


#dataset

class PetSegmentationDataset(Dataset):
    def __init__(self, root, split="trainval", img_size=128, download=True):
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            download=download,
        )
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_resize = transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_pil, mask_pil = self.base[idx]
        image    = self.img_transform(image_pil)
        mask_pil = self.mask_resize(mask_pil)
        mask_np  = np.array(mask_pil, dtype=np.int32)
        binary   = (mask_np == 1).astype(np.float32)
        mask     = torch.from_numpy(binary).unsqueeze(0)
        return image, mask


def get_dataloaders(cfg):
    full_ds = PetSegmentationDataset(
        root=cfg["data_root"],
        img_size=cfg["img_size"],
        download=cfg["download"],
    )
    if cfg.get("subset_size") and cfg["subset_size"] < len(full_ds):
        indices = list(range(cfg["subset_size"]))
        full_ds = Subset(full_ds, indices)

    total      = len(full_ds)
    val_size   = int(total * cfg["val_split"])
    train_size = total - val_size

    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True, num_workers=cfg["num_workers"], pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=False,
    )
    print(f"[Dataset]  Total={total}  |  Train={train_size}  |  Val={val_size}")
    return train_loader, val_loader


#U-net

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        features = self.conv(x)
        pooled   = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch * 2, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1      = EncoderBlock(in_channels, 32)
        self.enc2      = EncoderBlock(32,  64)
        self.enc3      = EncoderBlock(64,  128)
        self.bottleneck = ConvBlock(128, 256)
        self.dec3      = DecoderBlock(256, 128)
        self.dec2      = DecoderBlock(128, 64)
        self.dec1      = DecoderBlock(64,  32)
        self.out       = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        x     = self.bottleneck(x)
        x     = self.dec3(x, s3)
        x     = self.dec2(x, s2)
        x     = self.dec1(x, s1)
        return self.out(x)


#Loss

def dice_loss(logits, targets, smooth=1.0):
    probs        = torch.sigmoid(logits)
    probs_flat   = probs.reshape(-1)
    targets_flat = targets.reshape(-1)
    intersection = (probs_flat * targets_flat).sum()
    dice         = (2.0 * intersection + smooth) / (
                       probs_flat.sum() + targets_flat.sum() + smooth)
    return 1.0 - dice


def combined_loss(logits, targets):
    bce  = nn.BCEWithLogitsLoss()(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


#Evaluating our model

@torch.no_grad()
def pixel_accuracy(logits, targets, threshold=0.5):
    preds   = (torch.sigmoid(logits) > threshold).float()
    correct = (preds == targets).float().sum()
    return (correct / targets.numel()).item()


@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds        = (torch.sigmoid(logits) > threshold).float()
    p, t         = preds.reshape(-1), targets.reshape(-1)
    intersection = (p * t).sum()
    union        = p.sum() + t.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


@torch.no_grad()
def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds        = (torch.sigmoid(logits) > threshold).float()
    p, t         = preds.reshape(-1), targets.reshape(-1)
    intersection = (p * t).sum()
    return ((2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)).item()


def compute_metrics(logits, targets):
    return {
        "pixel_acc": pixel_accuracy(logits, targets),
        "iou"      : iou_score(logits, targets),
        "dice"     : dice_score(logits, targets),
    }


#Training

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


#Validation

def validate(model, loader, device):
    model.eval()
    totals = {"loss": 0.0, "pixel_acc": 0.0, "iou": 0.0, "dice": 0.0}
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            totals["loss"] += combined_loss(logits, masks).item()
            for k, v in compute_metrics(logits, masks).items():
                totals[k] += v
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


#Vizualization

def plot_training_curves(train_losses, val_losses, val_history):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2196F3", lw=2)
    axes[0].plot(epochs, val_losses,   label="Val Loss",   color="#FF5722", lw=2, ls="--")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    axes[1].plot(epochs, [m["pixel_acc"] for m in val_history], label="Pixel Accuracy", lw=2)
    axes[1].plot(epochs, [m["iou"]       for m in val_history], label="IoU",            lw=2)
    axes[1].plot(epochs, [m["dice"]      for m in val_history], label="Dice Score",     lw=2)
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation Metrics")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("Results.png", dpi=150, bbox_inches="tight")
    plt.close()


def visualize_predictions(model, loader, device, num_samples=20):
    model.eval()
    images, masks = next(iter(loader))
    with torch.no_grad():
        preds = (torch.sigmoid(model(images.to(device))) > 0.5).float().cpu()

    images = images.numpy()
    masks  = masks.numpy()
    preds  = preds.numpy()

    n = min(num_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3.2))
    fig.suptitle("Segmentation Predictions", fontsize=14, fontweight="bold", y=1.01)
    for col, title in enumerate(["Input Image", "Ground Truth", "Prediction"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for i in range(n):
        img = np.clip(np.transpose(images[i], (1, 2, 0)), 0, 1)
        axes[i, 0].imshow(img);                              axes[i, 0].axis("off")
        axes[i, 1].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1); axes[i, 1].axis("off")
        axes[i, 2].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1); axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
    plt.close()


#The moment of truth
def predict_single_image(image_path: str, model_path: str, img_size: int = 128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    if not os.path.exists(model_path):
        print(f"\nNo model: {model_path}")
        print("Add a model first")
        return

    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if not os.path.exists(image_path):
        print(f"no image: {image_path}")
        return

    original_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_pil.size

    img_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img_tensor = img_transform(original_pil).unsqueeze(0).to(device)  

    with torch.no_grad():
        logit = model(img_tensor)
        prob  = torch.sigmoid(logit).squeeze().cpu().numpy()   
        mask  = (prob > 0.5).astype(np.uint8)                  

    pet_confidence = float(prob[mask == 1].mean()) if mask.sum() > 0 else 0.0
    pet_ratio      = float(mask.mean())            

    orig_np = np.array(original_pil.resize((img_size, img_size))).astype(np.float32) / 255.0
    overlay = orig_np.copy()

    overlay[mask == 1, 0] = overlay[mask == 1, 0] * 0.4
    overlay[mask == 1, 1] = np.clip(overlay[mask == 1, 1] * 0.6 + 0.4, 0, 1)
    overlay[mask == 1, 2] = overlay[mask == 1, 2] * 0.4

    
    if pet_ratio > 0.05 and pet_confidence > 0.6:
        verdict = f"It's a pet (Confidence: {pet_confidence:.0%},  Ratio: {pet_ratio:.0%})"
    elif pet_ratio > 0.02:
        verdict = f"Maybe a pet (Confidence: {pet_confidence:.0%},  Ratio: {pet_ratio:.0%})"
    else:
        verdict = f"No pet detected, sorry  (Ratio: {pet_ratio:.0%})"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(verdict, fontsize=13, fontweight="bold", color="#1a1a2e", y=1.02)

    titles = ["Original", "Mask", "Overlay"]
    imgs   = [orig_np, prob, overlay]
    cmaps  = [None, "RdYlGn", None]

    for ax, title, img, cmap in zip(axes, titles, imgs, cmaps):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.axis("off")

    plt.tight_layout()
    out_path = "test_result.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*55}")
    print(f"  {verdict}")
    print(f"  Resolution: {orig_w}×{orig_h}")
    print(f"  Model resolution:     {img_size}×{img_size}")
    print(f"  Saved →        {out_path}")
    print(f"{'='*55}\n")


#main

def main():
    check_dependencies()

    class Args:
        train = True
        test = "testimage.png"
        model = "unet_pet_segmentation.pth"

    args = Args()

    cfg = FULL_CONFIG if args.train else FAST_CONFIG
    device = cfg["device"]

    mode_label = "Full train" if args.train else "Fast mode"
    print(f"\n{'='*60}")
    print(f"  U-Net  |  Oxford-IIIT Pet  |  {mode_label}")
    print(f"{'='*60}")
    print(f"  Device    : {device.upper()}")
    print(f"  Epochs    : {cfg['epochs']}")
    print(f"  LR        : {cfg['lr']}")
    print(f"  Batch     : {cfg['batch_size']}")
    print(f"  ImgSize   : {cfg['img_size']}×{cfg['img_size']}")
    if cfg.get("subset_size"):
        print(f"  Subset    : {cfg['subset_size']}")
    print(f"{'='*60}\n")

    

    print("Loading dataset")
    train_loader, val_loader = get_dataloaders(cfg)

    model     = UNet(in_channels=3, out_channels=1).to(device)
    params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]   Trainable parameters: {params:,}\n")

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    train_losses, val_losses, val_history = [], [], []

    header = (f"{'Epoch':>8}  {'Train Loss':>10}  {'Val Loss':>9}  "
              f"{'Pixel Acc':>10}  {'IoU':>8}  {'Dice':>8}")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_res = validate(model, val_loader, device)

        train_losses.append(tr_loss)
        val_losses.append(val_res["loss"])
        val_history.append(val_res)

        print(f"[{epoch:02d}/{cfg['epochs']:02d}]    "
              f"{tr_loss:10.4f}  "
              f"{val_res['loss']:9.4f}  "
              f"{val_res['pixel_acc']:10.4f}  "
              f"{val_res['iou']:8.4f}  "
              f"{val_res['dice']:8.4f}")

    print("─" * len(header))

    torch.save(model.state_dict(), cfg["save_path"])
    print(f"\n[Saved]   Model → {cfg['save_path']}")

    print("\nVizualization")
    plot_training_curves(train_losses, val_losses, val_history)
    visualize_predictions(model, val_loader, device, num_samples=4)

    f = val_history[-1]
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Pixel Accuracy : {f['pixel_acc']:.4f}")
    print(f"  IoU Score      : {f['iou']:.4f}")
    print(f"  Dice Score     : {f['dice']:.4f}")
    print(f"{'='*60}")
    print(f"Done\n")

    if args.test:
        predict_single_image(args.test, args.model, img_size=cfg["img_size"])

if __name__ == "__main__":
    main()
