#!/usr/bin/env python3
"""
Fine-tune DDPM UNet with boundary-aware loss from a frozen WRN classifier.
This version uses per-class Gaussian Mixture Models (GMM) on classifier logits
instead of k-means to define cluster means ("centroids").
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image
from typing import Dict, Tuple

# Try to import sklearn's GaussianMixture for CPU-side clustering
try:
    from sklearn.mixture import GaussianMixture
    _HAS_SKLEARN = True
except Exception as _e:
    GaussianMixture = None
    _HAS_SKLEARN = False

# -------------------------
# Config (edit as needed)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFIER_ARCH = "wrn-28-10"
NUM_CLASSES = 10
CLASSIFIER_CKPT = "/home/c01sogh/CISPA-home/trades/TRADES-master/cifar10_20percent_dataratio0.3generated_DDPM_beta60percentrandomsamples/model-wideres-epoch200.pt"

DIFFUSION_ID = "google/ddpm-cifar10-32"

DATA_ROOT = "./data"
BATCH_SIZE_TRAIN = 64
NUM_WORKERS = 4

# GMM / boundary config
K_PER_CLASS = 5              # number of Gaussian components per class
GMM_MAX_ITERS = 100          # EM iterations (sklearn)
GMM_COVARIANCE_TYPE = "diag" # {"full","tied","diag","spherical"}
GMM_REG_COVAR = 1e-6         # numerical stability
EMBED_WITH_LOGITS = True     # use classifier logits as embedding

# Fine-tune config
LR = 2e-5
N_EPOCHS = 5
LAMBDA_BOUNDARY = 0.5      # weight for boundary loss
TIMESTEPS_TRAIN = None     # If you'd like fewer timesteps for speed, set int

# Sampling config
SAMPLES_OUT_DIR = "/home/c01sogh/CISPA-az6/dropattack-2024/YOLO_project_2/trades/newdiff_7_gmmepoch5"
TOTAL_IMAGES = 1000
SAMPLE_BATCH = 100
TIMESTEPS_SAMPLE = 1000    # sampling timesteps (int) or None to use scheduler default

FINETUNED_OUT = "/home/c01sogh/CISPA-home/trades/TRADES-master/fine_tuned_ddpm_cluster_boundary_gmm.pth"
os.makedirs(os.path.dirname(FINETUNED_OUT), exist_ok=True)
os.makedirs(SAMPLES_OUT_DIR, exist_ok=True)

# -------------------------
# Local imports (your code)
# -------------------------
# Ensure your project path is on PYTHONPATH or run script from repo root
from utils import get_model  # your WRN loader - must return nn.Module

# -------------------------
# 1) Load frozen classifier
# -------------------------
print("🔒 Loading frozen classifier...")
classifier = get_model(CLASSIFIER_ARCH, num_classes=NUM_CLASSES, normalize_input=False)

# load checkpoint safely (handle state_dict vs saved model)
ckpt = torch.load(CLASSIFIER_CKPT, map_location="cpu")
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

def strip_module_prefix(state_dict_in):
    new = {}
    for k, v in state_dict_in.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new[new_key] = v
    return new

try:
    classifier.load_state_dict(state_dict)
except RuntimeError:
    print("⚠️ Direct load_state_dict failed; attempting to strip 'module.' prefixes and retry...")
    classifier.load_state_dict(strip_module_prefix(state_dict))

classifier = nn.DataParallel(classifier).to(device)
classifier.eval()
for p in classifier.parameters():
    p.requires_grad = False
print("✅ Classifier loaded & frozen.")

# -------------------------
# 2) Load diffusion UNet + scheduler
# -------------------------
print(" Loading diffusion model & scheduler...")
unet = UNet2DModel.from_pretrained(DIFFUSION_ID).to(device)
unet.train()
scheduler = DDPMScheduler.from_pretrained(DIFFUSION_ID)

if TIMESTEPS_TRAIN is not None:
    scheduler.set_timesteps(TIMESTEPS_TRAIN)

# -------------------------
# 3) CIFAR-10 dataloaders
# -------------------------
print(" Preparing CIFAR-10 dataloaders...")
# DDPM expects inputs in [-1, 1]. Use 3-channel normalization.
ddpm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=ddpm_transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=NUM_WORKERS)
embed_loader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)

# -------------------------
# 4) Build embeddings + pseudo-labels using classifier logits
# -------------------------
@torch.no_grad()
def build_embeddings_and_pseudolabels(loader: DataLoader, model: nn.Module, use_logits: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_embeds = []
    all_yhat = []
    for xb, _ in tqdm(loader, desc=" Embedding CIFAR-10 with classifier"):
        xb = xb.to(device)
        logits = model(xb)  # (B, NUM_CLASSES)
        yhat = logits.argmax(dim=1)
        embeds = logits if use_logits else logits  # fallback to logits if no penultimate features
        all_embeds.append(embeds.detach().cpu())
        all_yhat.append(yhat.detach().cpu())
    Z = torch.cat(all_embeds, dim=0)
    yhat = torch.cat(all_yhat, dim=0)
    return Z, yhat

Z_all, yhat_all = build_embeddings_and_pseudolabels(embed_loader, classifier, use_logits=EMBED_WITH_LOGITS)
print(f" Embedding bank shape: {Z_all.shape}")

# -------------------------
# 5) Fit per-class GMMs on embeddings (CPU) and use component means as centroids
# -------------------------
if not _HAS_SKLEARN:
    raise ImportError(
        "scikit-learn is required for GMM clustering. Please install it via 'pip install scikit-learn'"
    )

print(" Fitting per-class Gaussian Mixture Models on embeddings (CPU)...")
class_centroids: Dict[int, torch.Tensor] = {}
Z_cpu = Z_all.float().cpu()
y_cpu = yhat_all.long().cpu()

for c in range(NUM_CLASSES):
    idx = (y_cpu == c).nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        # No samples predicted as class c; fall back to global mean
        z_mean = Z_cpu.mean(dim=0, keepdim=True)
        centroids = z_mean.repeat(K_PER_CLASS, 1)
        class_centroids[c] = centroids
        continue

    Zc = Z_cpu[idx]  # (Nc, D)

    if Zc.size(0) < K_PER_CLASS:
        # Not enough samples for requested components; repeat class mean
        cmean = Zc.mean(dim=0, keepdim=True)
        centroids = cmean.repeat(K_PER_CLASS, 1)
        class_centroids[c] = centroids
        continue

    # Fit GMM on numpy (sklearn expects numpy arrays)
    X = Zc.numpy()
    try:
        gmm = GaussianMixture(
            n_components=K_PER_CLASS,
            covariance_type=GMM_COVARIANCE_TYPE,
            max_iter=GMM_MAX_ITERS,
            reg_covar=GMM_REG_COVAR,
            random_state=42,
            init_params="kmeans",
        )
        gmm.fit(X)
        means = torch.from_numpy(gmm.means_).float()  # (K, D)
    except Exception as e:
        print(f"⚠️ GMM failed for class {c} with error: {e}. Falling back to repeated class mean.")
        cmean = Zc.mean(dim=0, keepdim=True)
        means = cmean.repeat(K_PER_CLASS, 1)

    class_centroids[c] = means

print(" GMM fitting complete.")

# Move centroids to device for fast distance computations
centroids_stack = {c: class_centroids[c].to(device) for c in range(NUM_CLASSES)}

# -------------------------
# Helper losses
# -------------------------
def confidence_minimization_loss(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    max_conf = probs.max(dim=1)[0]
    return -max_conf.mean()

@torch.no_grad()
def logits_and_pseudolabels(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = classifier(x)               # (B, NUM_CLASSES)
    yhat = logits.argmax(dim=1)          # (B,)
    return logits, yhat

# Note: we still compute Euclidean distances to GMM component means.
# Optionally, you could switch to Mahalanobis using each component's covariance.
def boundary_margin_loss_from_logits(logits: torch.Tensor, yhat: torch.Tensor, centroids_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
    embeds = logits
    B = embeds.size(0)
    margins = []
    for i in range(B):
        c = int(yhat[i].item())
        Cc = centroids_dict[c]  # (K, D)
        d = torch.cdist(embeds[i].unsqueeze(0), Cc, p=2).squeeze(0)
        top2, _ = torch.topk(d, k=min(2, d.numel()), largest=False)
        if top2.numel() == 1:
            margin = torch.tensor(0.0, device=embeds.device)
        else:
            margin = torch.abs(top2[0] - top2[1])
        margins.append(margin)
    if len(margins) == 0:
        return torch.tensor(0.0, device=embeds.device)
    return torch.stack(margins, dim=0).mean()

# -------------------------
# 7) Fine-tuning loop (DDPM + boundary margin loss)
# -------------------------
print(" Starting fine-tuning with boundary-aware loss (GMM centroids)...")
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

for epoch in range(N_EPOCHS):
    pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
    for xb, _ in pbar:
        xb = xb.to(device)
        bsz = xb.size(0)

        # Use a single (random) timestep per batch to avoid ambiguous boolean error
        t_int = int(torch.randint(0, scheduler.config.num_train_timesteps, (1,)).item())
        t_tensor = torch.full((bsz,), t_int, device=device, dtype=torch.long)

        # Add forward noise
        noise = torch.randn_like(xb)
        x_noisy = scheduler.add_noise(xb, noise, t_tensor)

        # Predict noise with unet (t passed as tensor of shape (B,))
        noise_pred = unet(x_noisy, t_tensor).sample

        # Standard DDPM loss
        ddpm_loss = F.mse_loss(noise_pred, noise)

        # One reverse step (batch) to get x_prev (x_{t-1}) estimate
        with torch.no_grad():
            out = scheduler.step(noise_pred, t_int, x_noisy)
            x_prev = out["prev_sample"]

        # Boundary loss computed in classifier embedding space
        logits, yhat = logits_and_pseudolabels(x_prev)
        boundary_loss = boundary_margin_loss_from_logits(logits, yhat, centroids_stack)

        loss = ddpm_loss + LAMBDA_BOUNDARY * boundary_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({
            "ddpm": f"{ddpm_loss.item():.4f}",
            "boundary": f"{boundary_loss.item():.4f}",
            "total": f"{loss.item():.4f}"
        })

print(" Fine-tuning complete.")

# -------------------------
# 8) Save fine-tuned UNet
# -------------------------
torch.save(unet.state_dict(), FINETUNED_OUT)
print(f" Saved fine-tuned DDPM UNet state_dict to: {FINETUNED_OUT}")

# -------------------------
# 9) Generate TOTAL_IMAGES samples (batched)
# -------------------------
print(f"🎨 Generating {TOTAL_IMAGES} boundary-seeking samples in batches...")
unet.eval()

# Ensure scheduler timesteps used for sampling are set (e.g., 1000)
if TIMESTEPS_SAMPLE is not None:
    scheduler.set_timesteps(TIMESTEPS_SAMPLE)

num_batches = 1000
img_idx = 0

with torch.no_grad():
    for b in range(num_batches):
        cur_batch = min(SAMPLE_BATCH, TOTAL_IMAGES - img_idx)
        x = torch.randn((cur_batch, 3, 32, 32), device=device)

        # reverse diffusion loop
        for t in scheduler.timesteps:
            t_int = int(t)
            t_tensor = torch.full((cur_batch,), t_int, device=device, dtype=torch.long)

            noise_pred = unet(x, t_tensor).sample
            step_out = scheduler.step(noise_pred, t_int, x)
            x = step_out["prev_sample"]

        # map to [0,1]
        samples = (x.clamp(-1, 1) + 1) / 2.0

        # save to disk
        for i, img_tensor in enumerate(samples):
            img = TF.to_pil_image(img_tensor.cpu())
            img.save(os.path.join(SAMPLES_OUT_DIR, f"sample_{img_idx:06d}.png"))
            img_idx += 1

        print(f" Saved batch {b+1}/{num_batches} (total saved: {img_idx})")

print(f" All {img_idx} images saved to: {SAMPLES_OUT_DIR}")
