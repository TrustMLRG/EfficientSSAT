from __future__ import annotations

import argparse
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import load_model_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 PGD Attack Evaluation")

    parser.add_argument(
        "--mode",
        default="whitebox",
        choices=("whitebox", "blackbox"),
        help="Attack mode.",
    )
    parser.add_argument("--model", default="wrn-28-10", help="Model architecture.")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Checkpoint path for white-box evaluation.",
    )
    parser.add_argument(
        "--source_model_path",
        default=None,
        help="Checkpoint path for the source model (black-box).",
    )
    parser.add_argument(
        "--target_model_path",
        default=None,
        help="Checkpoint path for the target model (black-box).",
    )

    parser.add_argument("--data_dir", default="./data", help="Dataset directory.")
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

    parser.add_argument("--batch_size", type=int, default=200, help="Batch size.")
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Optional cap on number of test examples to evaluate.",
    )

    parser.add_argument("--epsilon", type=float, default=0.031, help="Linf epsilon.")
    parser.add_argument("--num_steps", type=int, default=40, help="PGD steps.")
    parser.add_argument("--step_size", type=float, default=0.01, help="PGD step size.")
    parser.add_argument(
        "--random_start",
        dest="random_start",
        action="store_true",
        help="Use random initialization (default).",
    )
    parser.add_argument(
        "--no-random_start",
        dest="random_start",
        action="store_false",
        help="Disable random initialization.",
    )
    parser.set_defaults(random_start=True)

    parser.add_argument("--no-cuda", action="store_true", default=False, help="Disable CUDA.")
    return parser


def _die(message: str) -> None:
    raise SystemExit(message)


def _load_cifar10_test_loader(args: argparse.Namespace, use_cuda: bool) -> DataLoader:
    transform_test = transforms.Compose([transforms.ToTensor()])
    try:
        testset = datasets.CIFAR10(
            root=args.data_dir,
            train=False,
            download=args.download,
            transform=transform_test,
        )
    except Exception as e:  # noqa: BLE001
        _die(str(e))

    loader_kwargs = {"num_workers": 0, "pin_memory": True} if use_cuda else {"num_workers": 0}
    return DataLoader(testset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)


def _load_model(arch: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = load_model_from_checkpoint(
        arch,
        ckpt_path,
        map_location="cpu",
        num_classes=10,
        device=device,
    )
    model.eval()
    return model


def pgd_linf(
    model_for_grad: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    num_steps: int,
    step_size: float,
    random_start: bool,
) -> torch.Tensor:
    x_adv = x.detach()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model_for_grad(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


@torch.no_grad()
def _eval_batch(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    pred = model(x).argmax(dim=1)
    return pred.eq(y).sum().item()


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_loader = _load_cifar10_test_loader(args, use_cuda=use_cuda)

    if args.mode == "whitebox":
        if not args.model_path:
            _die("--model_path is required for whitebox mode.")
        model = _load_model(args.model, args.model_path, device)
        grad_model = model
        eval_model = model
    else:
        if not args.source_model_path or not args.target_model_path:
            _die("--source_model_path and --target_model_path are required for blackbox mode.")
        grad_model = _load_model(args.model, args.source_model_path, device)
        eval_model = _load_model(args.model, args.target_model_path, device)

    total = 0
    natural_correct = 0
    robust_correct = 0

    for data, target in test_loader:
        if args.n_examples is not None and total >= args.n_examples:
            break
        data, target = data.to(device), target.to(device)

        if args.n_examples is not None:
            remaining = args.n_examples - total
            if remaining <= 0:
                break
            if data.size(0) > remaining:
                data = data[:remaining]
                target = target[:remaining]

        natural_correct += _eval_batch(eval_model, data, target)
        x_adv = pgd_linf(
            grad_model,
            data,
            target,
            epsilon=float(args.epsilon),
            num_steps=int(args.num_steps),
            step_size=float(args.step_size),
            random_start=bool(args.random_start),
        )
        robust_correct += _eval_batch(eval_model, x_adv, target)

        total += int(target.size(0))

    natural_acc = 100.0 * natural_correct / max(1, total)
    robust_acc = 100.0 * robust_correct / max(1, total)

    print(f"Mode: {args.mode}")
    print(f"Natural Accuracy: {natural_acc:.2f}% ({natural_correct}/{total})")
    print(f"Robust Accuracy:  {robust_acc:.2f}% ({robust_correct}/{total})")


if __name__ == "__main__":
    main()
