from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from utils import load_model_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a CIFAR-10 DDPM UNet with a GMM-based boundary margin loss."
    )

    parser.add_argument(
        "--classifier_arch",
        default="wrn-28-10",
        help="Classifier architecture (see utils.get_model).",
    )
    parser.add_argument(
        "--classifier_ckpt",
        required=True,
        help="Path to classifier checkpoint (state_dict).",
    )

    parser.add_argument(
        "--diffusion_id",
        default="google/ddpm-cifar10-32",
        help="Diffusers model id or local path.",
    )

    parser.add_argument("--data_root", default="./data", help="Dataset directory.")
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="Download dataset if missing.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not download dataset (default).",
    )
    parser.set_defaults(download=False)

    # Embedding/GMM config
    parser.add_argument(
        "--embed_n_examples",
        type=int,
        default=5000,
        help="Number of training examples used to build the embedding bank for GMM fitting.",
    )
    parser.add_argument("--embed_batch_size", type=int, default=256)
    parser.add_argument("--k_per_class", type=int, default=5)
    parser.add_argument("--gmm_max_iters", type=int, default=100)
    parser.add_argument(
        "--gmm_covariance_type",
        type=str,
        default="diag",
        choices=("full", "tied", "diag", "spherical"),
    )
    parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)

    # Fine-tune config
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lambda_boundary", type=float, default=0.5)
    parser.add_argument(
        "--timesteps_train",
        type=int,
        default=None,
        help="Optional number of diffusion timesteps for training speed.",
    )

    # Outputs
    parser.add_argument(
        "--finetuned_out",
        default="./outputs/diffusionmgmm/finetuned.pth",
        help="Output path for fine-tuned UNet state_dict.",
    )
    parser.add_argument("--total_images", type=int, default=100)
    parser.add_argument("--sample_batch", type=int, default=50)
    parser.add_argument("--timesteps_sample", type=int, default=1000)
    parser.add_argument(
        "--samples_out",
        default="./outputs/diffusionmgmm/samples",
        help="Directory to write generated sample images.",
    )

    parser.add_argument("--no-cuda", action="store_true", default=False)
    return parser


def _die(message: str) -> None:
    raise SystemExit(message)


def _sample_train_timesteps(scheduler, batch_size: int, device: torch.device) -> torch.Tensor:
    timesteps = getattr(scheduler, "timesteps", None)
    if timesteps is None:
        return torch.randint(
            0, int(scheduler.config.num_train_timesteps), (batch_size,), device=device
        ).long()

    timesteps = torch.as_tensor(timesteps, device=device)
    indices = torch.randint(0, timesteps.numel(), (batch_size,), device=device)
    return timesteps[indices].long()


def _guidance_sample_from_step_output(step_output) -> torch.Tensor:
    pred_original_sample = getattr(step_output, "pred_original_sample", None)
    if pred_original_sample is not None:
        return pred_original_sample
    if isinstance(step_output, tuple) and len(step_output) >= 2 and step_output[1] is not None:
        return step_output[1]
    raise RuntimeError(
        "Scheduler.step did not return pred_original_sample; please use a diffusers version "
        "that exposes the clean reconstruction for guidance."
    )


def _predict_clean_sample(
    scheduler,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    noisy_sample: torch.Tensor,
) -> torch.Tensor:
    try:
        step_output = scheduler.step(noise_pred, timesteps, noisy_sample)
        return _guidance_sample_from_step_output(step_output)
    except Exception:
        parts = []
        for i in range(noisy_sample.size(0)):
            step_output = scheduler.step(
                noise_pred[i : i + 1], int(timesteps[i].item()), noisy_sample[i : i + 1]
            )
            parts.append(_guidance_sample_from_step_output(step_output))
        return torch.cat(parts, dim=0)


@torch.no_grad()
def _collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_examples: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits_all = []
    yhat_all = []
    seen = 0
    for x, _ in loader:
        if seen >= max_examples:
            break
        remaining = max_examples - seen
        if x.size(0) > remaining:
            x = x[:remaining]
        x = x.to(device)
        logits = model(x)
        yhat = logits.argmax(dim=1)
        logits_all.append(logits.detach().cpu())
        yhat_all.append(yhat.detach().cpu())
        seen += int(x.size(0))
    if not logits_all:
        raise ValueError("No data collected for embeddings.")
    logits_np = torch.cat(logits_all, dim=0).numpy()
    yhat_np = torch.cat(yhat_all, dim=0).numpy()
    return logits_np, yhat_np


def _fit_class_centroids(
    logits_np: np.ndarray,
    yhat_np: np.ndarray,
    *,
    k_per_class: int,
    gmm_cls,
    covariance_type: str,
    max_iter: int,
    reg_covar: float,
) -> dict[int, np.ndarray]:
    if k_per_class <= 0:
        raise ValueError("--k_per_class must be positive.")

    centroids: dict[int, np.ndarray] = {}
    n_classes = int(logits_np.shape[1])
    for c in range(n_classes):
        idx = np.where(yhat_np == c)[0]
        if idx.size == 0:
            mean = logits_np.mean(axis=0, keepdims=True)
            centroids[c] = np.repeat(mean, k_per_class, axis=0)
            continue

        class_logits = logits_np[idx]
        if class_logits.shape[0] < max(2, k_per_class):
            mean = class_logits.mean(axis=0, keepdims=True)
            centroids[c] = np.repeat(mean, k_per_class, axis=0)
            continue

        gmm = gmm_cls(
            n_components=k_per_class,
            covariance_type=str(covariance_type),
            max_iter=int(max_iter),
            reg_covar=float(reg_covar),
            random_state=42,
            init_params="kmeans",
        )
        gmm.fit(class_logits)
        centroids[c] = gmm.means_.astype(np.float32)

    return centroids


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    try:
        from diffusers import UNet2DModel, DDPMScheduler  # type: ignore
        from tqdm import tqdm  # type: ignore
    except Exception as e:  # noqa: BLE001
        _die(f"diffusers and tqdm are required. Install with: pip install -r requirements.txt\n{e}")

    try:
        from sklearn.mixture import GaussianMixture  # type: ignore
    except Exception as e:  # noqa: BLE001
        _die(f"scikit-learn is required. Install with: pip install -r requirements.txt\n{e}")

    # Load frozen classifier (expects inputs in [0, 1]).
    classifier = load_model_from_checkpoint(
        args.classifier_arch,
        args.classifier_ckpt,
        map_location="cpu",
        num_classes=10,
        device=device,
    )
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # Prepare embedding loader (CIFAR-10 in [0,1]).
    embed_transform = transforms.Compose([transforms.ToTensor()])
    try:
        embed_set = datasets.CIFAR10(
            root=args.data_root, train=True, download=args.download, transform=embed_transform
        )
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    embed_loader = DataLoader(
        embed_set,
        batch_size=int(args.embed_batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
    )

    logits_np, yhat_np = _collect_logits(
        classifier, embed_loader, device, max_examples=int(args.embed_n_examples)
    )

    # Fit per-class GMMs on logits (CPU), use means as centroids.
    try:
        centroids = _fit_class_centroids(
            logits_np,
            yhat_np,
            k_per_class=int(args.k_per_class),
            gmm_cls=GaussianMixture,
            covariance_type=str(args.gmm_covariance_type),
            max_iter=int(args.gmm_max_iters),
            reg_covar=float(args.gmm_reg_covar),
        )
    except ValueError as e:
        _die(str(e))

    centroids_t = {c: torch.from_numpy(m).to(device) for c, m in centroids.items()}

    # Load diffusion UNet + scheduler.
    unet = UNet2DModel.from_pretrained(args.diffusion_id).to(device)
    unet.train()
    scheduler = DDPMScheduler.from_pretrained(args.diffusion_id)
    if args.timesteps_train is not None:
        scheduler.set_timesteps(int(args.timesteps_train))

    # CIFAR-10 train loader in [-1,1] for DDPM.
    ddpm_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    try:
        trainset = datasets.CIFAR10(
            root=args.data_root, train=True, download=args.download, transform=ddpm_transform
        )
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    trainloader = DataLoader(
        trainset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=use_cuda,
    )

    def boundary_margin_loss(logits: torch.Tensor) -> torch.Tensor:
        yhat = logits.argmax(dim=1)
        margins = []
        for i in range(logits.size(0)):
            c = int(yhat[i].item())
            Cc = centroids_t[c]  # (K, D)
            d = torch.cdist(logits[i].unsqueeze(0), Cc, p=2).squeeze(0)  # (K,)
            k = min(2, d.numel())
            topk, _ = torch.topk(d, k=k, largest=False)
            if topk.numel() < 2:
                margins.append(torch.tensor(0.0, device=device))
            else:
                margins.append(torch.abs(topk[0] - topk[1]))
        return torch.stack(margins, dim=0).mean()

    optimizer = torch.optim.Adam(unet.parameters(), lr=float(args.lr))

    for epoch in range(int(args.epochs)):
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{int(args.epochs)}")
        for x_start, _ in pbar:
            x_start = x_start.to(device)
            bsz = x_start.size(0)
            t = _sample_train_timesteps(scheduler, bsz, device)

            noise = torch.randn_like(x_start)
            x_noisy = scheduler.add_noise(x_start, noise, t)
            noise_pred = unet(x_noisy, t).sample
            ddpm_loss = F.mse_loss(noise_pred, noise)

            x_recon = _predict_clean_sample(scheduler, noise_pred, t, x_noisy)

            x_for_clf = (x_recon.clamp(-1, 1) + 1) / 2
            logits = classifier(x_for_clf)
            b_loss = boundary_margin_loss(logits)

            loss = ddpm_loss + float(args.lambda_boundary) * b_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=float(loss.item()), ddpm=float(ddpm_loss.item()), boundary=float(b_loss.item()))

    finetuned_out = Path(args.finetuned_out)
    finetuned_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unet.state_dict(), finetuned_out)
    print(f"Saved fine-tuned UNet to: {finetuned_out}")

    # Sampling
    samples_out = Path(args.samples_out)
    samples_out.mkdir(parents=True, exist_ok=True)
    unet.eval()

    total_images = int(args.total_images)
    sample_batch = int(args.sample_batch)
    timesteps_sample = int(args.timesteps_sample)
    scheduler.set_timesteps(timesteps_sample)

    produced = 0
    idx = 0
    while produced < total_images:
        cur_bs = min(sample_batch, total_images - produced)
        x = torch.randn((cur_bs, 3, 32, 32), device=device)
        for t in scheduler.timesteps:
            with torch.no_grad():
                noise_pred = unet(x, t).sample
            x = scheduler.step(noise_pred, t, x).prev_sample
        samples = (x.clamp(-1, 1) + 1) / 2
        for i in range(cur_bs):
            img = TF.to_pil_image(samples[i].cpu())
            img.save(samples_out / f"sample_{idx:06d}.png")
            idx += 1
        produced += cur_bs

    print(f"Wrote {total_images} samples to: {samples_out}")


if __name__ == "__main__":
    main()
