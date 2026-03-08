from __future__ import annotations

import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import load_model_from_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CIFAR-10 AutoAttack Evaluation")

    parser.add_argument("--model", default="wrn-28-10", help="Model architecture.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Checkpoint path for evaluation.",
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
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.031,
        help="Attack epsilon (Linf).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="standard",
        help="AutoAttack version (e.g., standard).",
    )
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


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    try:
        from autoattack import AutoAttack  # type: ignore
    except Exception as e:  # noqa: BLE001
        _die(f"autoattack is required. Install dependencies with: pip install -r requirements.txt\n{e}")

    model = _load_model(args.model, args.model_path, device)
    test_loader = _load_cifar10_test_loader(args, use_cuda=use_cuda)

    x_list = []
    y_list = []
    seen = 0
    for x, y in test_loader:
        if args.n_examples is not None and seen >= args.n_examples:
            break
        if args.n_examples is not None:
            remaining = args.n_examples - seen
            x = x[:remaining]
            y = y[:remaining]
        x_list.append(x)
        y_list.append(y)
        seen += int(y.size(0))

    if not x_list:
        _die("No test data loaded.")

    x_test = torch.cat(x_list, dim=0).to(device)
    y_test = torch.cat(y_list, dim=0).to(device)

    with torch.no_grad():
        clean_pred = model(x_test).argmax(dim=1)
        clean_acc = 100.0 * clean_pred.eq(y_test).float().mean().item()

    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=float(args.epsilon),
        version=str(args.version),
        device="cuda" if use_cuda else "cpu",
    )

    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=int(args.batch_size))

    with torch.no_grad():
        adv_pred = model(x_adv).argmax(dim=1)
        robust_acc = 100.0 * adv_pred.eq(y_test).float().mean().item()

    print(f"Natural Accuracy: {clean_acc:.2f}%")
    print(f"Robust Accuracy:  {robust_acc:.2f}%")


if __name__ == "__main__":
    main()
