from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import DATASETS, SemiSupervisedDataset, SemiSupervisedSampler
from trades import trades_loss
from utils import get_model


def _model_supports_prelogit(name: str) -> bool:
    return name.startswith(("wrn-", "ss-"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PyTorch training script for EfficientSSAT and TRADES-based baselines."
    )

    # Dataset config
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=DATASETS,
        help="The dataset to use for training.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="Directory where datasets are located.",
    )
    parser.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="Download datasets if missing.",
    )
    parser.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Do not download datasets (default).",
    )
    parser.set_defaults(download=False)

    parser.add_argument(
        "--svhn_extra",
        action="store_true",
        default=False,
        help="Add the extra SVHN split as auxiliary data.",
    )

    # Model config
    parser.add_argument(
        "--model",
        "-m",
        default="wrn-28-10",
        type=str,
        help="Name of the model (see utils.get_model).",
    )
    parser.add_argument(
        "--model_dir",
        default="./checkpoints/run1",
        help="Directory for saving checkpoints and logs.",
    )
    parser.add_argument(
        "--normalize_input",
        action="store_true",
        default=False,
        help="Apply standard CIFAR normalization inside the model.",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="Number of batches between logging of training status.",
    )
    parser.add_argument(
        "--save_freq",
        default=25,
        type=int,
        help="Checkpoint save frequency (in epochs).",
    )

    # Generic training configs
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
        help="Input batch size for training.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=500,
        metavar="N",
        help="Input batch size for testing.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        metavar="N",
        help="Number of epochs to train.",
    )

    # Eval config
    parser.add_argument(
        "--eval_freq",
        default=1,
        type=int,
        help="Eval frequency (in epochs).",
    )
    parser.add_argument(
        "--train_eval_batches",
        default=None,
        type=int,
        help="Maximum number of batches in training set eval.",
    )

    # Optimizer config
    parser.add_argument("--weight_decay", "--wd", default=5e-4, type=float)
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR", help="Learning rate.")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default=None,
        choices=("stepwise", "cosine"),
        help="Learning-rate schedule. The paper uses stepwise decay; cosine is available for ablations.",
    )
    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        nargs="+",
        default=[75, 90, 100],
        help="Epochs to decay the LR for the stepwise schedule.",
    )
    parser.add_argument(
        "--lr_decay_factor",
        type=float,
        default=10.0,
        help="Decay factor for the stepwise schedule (e.g., 10 means lr *= 0.1 at each milestone).",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum.",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=None,
        help="Use Nesterov momentum.",
    )

    # Adversarial / stability training config
    parser.add_argument(
        "--distance",
        "-d",
        default="l_inf",
        type=str,
        choices=["l_inf", "l_2"],
        help="Attack metric.",
    )
    parser.add_argument(
        "--epsilon",
        default=0.031,
        type=float,
        help="Adversarial perturbation size.",
    )
    parser.add_argument(
        "--pgd_num_steps",
        default=10,
        type=int,
        help="Number of PGD steps in adversarial training.",
    )
    parser.add_argument(
        "--pgd_step_size",
        default=0.007,
        type=float,
        help="PGD step size in adversarial training.",
    )
    parser.add_argument(
        "--beta",
        default=6.0,
        type=float,
        help="TRADES regularization weight.",
    )

    # Auxiliary-data training configuration
    parser.add_argument(
        "--aux_data_filename",
        default=None,
        type=str,
        help="Path to aux data (.pickle/.pkl or .npz).",
    )
    parser.add_argument(
        "--generated_images_dir",
        default=None,
        type=str,
        help="Directory containing generated images (PNG/JPG).",
    )
    parser.add_argument(
        "--unsup_fraction",
        default=None,
        type=float,
        help=(
            "Fraction of unsupervised examples in each batch. "
            "If omitted, defaults to 0.5 when aux data is provided, else 0.0."
        ),
    )
    parser.add_argument(
        "--aux_take_amount",
        default=None,
        type=int,
        help="Optional cap on aux examples (after loading).",
    )
    parser.add_argument(
        "--remove_pseudo_labels",
        action="store_true",
        default=False,
        help="Train without pseudo-labels (treat aux labels as -1).",
    )

    # Aux selection configuration
    parser.add_argument(
        "--selection_method",
        default="none",
        choices=("none", "predconf", "lcs-km", "lcs-gmm"),
        help="Aux selection method.",
    )
    parser.add_argument(
        "--selection_model_arch",
        default=None,
        type=str,
        help="Model architecture used for aux selection/pseudo-labeling.",
    )
    parser.add_argument(
        "--selection_model_ckpt",
        default=None,
        type=str,
        help="Checkpoint path used for aux selection/pseudo-labeling.",
    )
    # Default selection sizes used in the CIFAR-10 ablation experiments.
    parser.add_argument("--n_boundary", default=20000, type=int)
    parser.add_argument("--n_random", default=30000, type=int)
    parser.add_argument("--selection_seed", default=42, type=int)
    parser.add_argument("--kmeans_k", default=10, type=int)
    parser.add_argument("--gmm_k", default=10, type=int)
    parser.add_argument("--selection_batch_size", default=256, type=int)
    parser.add_argument("--selection_num_workers", default=0, type=int)

    # Additional data augmentation
    parser.add_argument(
        "--autoaugment",
        action="store_true",
        default=False,
        help="Use torchvision AutoAugment with a dataset-specific policy.",
    )
    parser.add_argument(
        "--cutout",
        action="store_true",
        default=False,
        help="Use RandomErasing as a Cutout-like augmentation.",
    )

    parser.add_argument(
        "--carmon_compat",
        action="store_true",
        default=False,
        help=(
            "Use hyperparameter defaults closer to the semisup-adv (Carmon et al.) "
            "robust_self_training.py baseline (e.g., enable Nesterov and cosine LR)."
        ),
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disable CUDA.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Run a single synthetic optimization step and exit.",
    )

    return parser


def _configure_logging(model_dir: Path, args: argparse.Namespace) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(model_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Args: %s", args)


def _build_transforms(args: argparse.Namespace):
    train_tfms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if args.autoaugment:
        policy = (
            transforms.AutoAugmentPolicy.SVHN
            if args.dataset == "svhn"
            else transforms.AutoAugmentPolicy.CIFAR10
        )
        train_tfms.append(transforms.AutoAugment(policy))
    train_tfms.append(transforms.ToTensor())
    if args.cutout:
        train_tfms.append(
            transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
        )
    transform_train = transforms.Compose(train_tfms)
    transform_test = transforms.Compose([transforms.ToTensor()])
    return transform_train, transform_test


def _die(message: str) -> None:
    raise SystemExit(message)


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.lr_schedule is None:
        args.lr_schedule = "cosine" if args.carmon_compat else "stepwise"

    if args.nesterov is None:
        args.nesterov = bool(args.carmon_compat)

    if args.unsup_fraction is None:
        has_aux = (
            args.aux_data_filename is not None
            or args.generated_images_dir is not None
            or (args.dataset == "svhn" and args.svhn_extra)
        )
        args.unsup_fraction = 0.5 if has_aux else 0.0

    if args.selection_model_arch is None:
        args.selection_model_arch = args.model

    if args.selection_method in {"lcs-km", "lcs-gmm"} and not _model_supports_prelogit(
        args.selection_model_arch
    ):
        raise ValueError(
            "--selection_model_arch must expose prelogit features for "
            f"{args.selection_method}; got {args.selection_model_arch!r}."
        )

    if int(args.pgd_num_steps) <= 0:
        raise ValueError("--pgd_num_steps must be positive.")

    return args


def _make_dataloaders(args: argparse.Namespace, device: torch.device, use_cuda: bool):
    transform_train, transform_test = _build_transforms(args)
    loader_kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {"num_workers": 0}

    try:
        trainset = SemiSupervisedDataset(
            base_dataset=args.dataset,
            add_svhn_extra=args.svhn_extra,
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform_train,
            aux_data_filename=args.aux_data_filename,
            generated_images_dir=args.generated_images_dir,
            add_aux_labels=not args.remove_pseudo_labels,
            aux_take_amount=args.aux_take_amount,
            selection_method=args.selection_method,
            selection_model_arch=args.selection_model_arch,
            selection_model_ckpt=args.selection_model_ckpt,
            n_boundary=args.n_boundary,
            n_random=args.n_random,
            selection_seed=args.selection_seed,
            kmeans_k=args.kmeans_k,
            gmm_k=args.gmm_k,
            selection_batch_size=args.selection_batch_size,
            selection_num_workers=args.selection_num_workers,
            selection_device=device,
        )
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    train_batch_sampler = SemiSupervisedSampler(
        trainset.sup_indices,
        trainset.unsup_indices,
        args.batch_size,
        args.unsup_fraction,
        num_batches=int(np.ceil(50000 / args.batch_size)),
    )
    epoch_size = len(train_batch_sampler) * args.batch_size
    train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **loader_kwargs)

    try:
        testset = SemiSupervisedDataset(
            base_dataset=args.dataset,
            root=args.data_dir,
            train=False,
            download=args.download,
            transform=transform_test,
        )
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader, epoch_size


def _accuracy_and_loss(output: torch.Tensor, target: torch.Tensor) -> tuple[float, int, int]:
    mask = target >= 0
    if not torch.any(mask):
        return 0.0, 0, 0
    loss = F.cross_entropy(output[mask], target[mask], reduction="sum").item()
    pred = output[mask].argmax(dim=1)
    correct = pred.eq(target[mask]).sum().item()
    total = int(mask.sum().item())
    return loss, correct, total


def _build_lr_scheduler(
    args: argparse.Namespace, optimizer: optim.Optimizer
) -> Optional[optim.lr_scheduler._LRScheduler]:
    schedule = str(args.lr_schedule).lower()
    if schedule == "stepwise":
        return None

    if schedule == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs))

    raise ValueError(f"Unknown lr_schedule: {args.lr_schedule!r}")


def _adjust_stepwise_lr(args: argparse.Namespace, optimizer: optim.Optimizer, epoch: int) -> None:
    milestones = [int(e) for e in args.lr_decay_epochs]
    milestones = sorted({e for e in milestones if e > 0})
    if not milestones:
        return

    factor = float(args.lr_decay_factor)
    if factor <= 0:
        raise ValueError("--lr_decay_factor must be positive.")

    lr = float(args.lr)
    for e in milestones:
        if epoch >= e:
            lr /= factor

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_one_epoch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    epoch_size: int,
) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = trades_loss(
            model=model,
            x_natural=data,
            y=target,
            optimizer=optimizer,
            step_size=args.pgd_step_size,
            epsilon=args.epsilon,
            perturb_steps=args.pgd_num_steps,
            beta=args.beta,
            distance=args.distance,
        )
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            seen = batch_idx * len(data)
            total = epoch_size
            pct = 100.0 * batch_idx / max(1, (epoch_size // max(1, args.batch_size)))
            msg = f"Train Epoch: {epoch} [{seen}/{total} ({pct:.0f}%)]\tLoss: {loss.item():.6f}"
            print(msg)
            logging.info(msg)


@torch.no_grad()
def eval_loader(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    max_batches: Optional[int] = None,
    name: str = "Eval",
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch_idx, (data, target) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss, correct, count = _accuracy_and_loss(output, target)
        total_loss += loss
        total_correct += correct
        total_count += count

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    msg = f"{name}: Average loss: {avg_loss:.4f}, Accuracy: {total_correct}/{total_count} ({100.0 * acc:.0f}%)"
    print(msg)
    logging.info(msg)
    return avg_loss, acc


def run_dry_run(args: argparse.Namespace, device: torch.device) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_classes = 10
    model = get_model(args.model, num_classes=num_classes, normalize_input=args.normalize_input)
    model.to(device)
    model.train()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    bsz = min(int(args.batch_size), 32)
    x = torch.rand((bsz, 3, 32, 32), device=device)
    y = torch.randint(0, num_classes, (bsz,), device=device)

    optimizer.zero_grad(set_to_none=True)
    loss = trades_loss(
        model=model,
        x_natural=x,
        y=y,
        optimizer=optimizer,
        step_size=args.pgd_step_size,
        epsilon=args.epsilon,
        perturb_steps=int(args.pgd_num_steps),
        beta=args.beta,
        distance=args.distance,
    )
    loss.backward()
    optimizer.step()
    print(f"Dry-run OK. Loss: {loss.item():.6f}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args = _resolve_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dry_run:
        run_dry_run(args, device)
        return

    model_dir = Path(args.model_dir)
    _configure_logging(model_dir, args)

    train_loader, test_loader, epoch_size = _make_dataloaders(args, device, use_cuda)

    num_classes = 10
    try:
        model = get_model(args.model, num_classes=num_classes, normalize_input=args.normalize_input)
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )
    scheduler = _build_lr_scheduler(args, optimizer)

    for epoch in range(1, int(args.epochs) + 1):
        if str(args.lr_schedule).lower() == "stepwise":
            _adjust_stepwise_lr(args, optimizer, epoch)
        train_one_epoch(args, model, device, train_loader, optimizer, epoch, epoch_size)

        if epoch % int(args.eval_freq) == 0:
            print("=" * 64)
            eval_loader(
                model,
                device,
                train_loader,
                max_batches=args.train_eval_batches,
                name="Training",
            )
            eval_loader(model, device, test_loader, name="Test")
            print("=" * 64)

        if scheduler is not None:
            scheduler.step()

        if epoch % int(args.save_freq) == 0:
            torch.save(model.state_dict(), model_dir / f"model-epoch{epoch}.pt")
            torch.save(optimizer.state_dict(), model_dir / f"opt-epoch{epoch}.pt")


if __name__ == "__main__":
    main()
