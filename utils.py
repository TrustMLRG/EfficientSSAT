"""
Utilities. Partially based on code from
https://github.com/modestyachts/CIFAR-10.1
"""
from __future__ import annotations

import json
import os
import pickle

import pathlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from models.wideresnet import WideResNet
from models.shake_shake import ShakeNet
from models.cifar_resnet import ResNet

from torch.nn import Sequential, Module

cifar10_label_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_model(name, num_classes=10, normalize_input=False):
    name_parts = name.split("-")
    if name_parts[0] == "wrn":
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = WideResNet(depth=depth, num_classes=num_classes, widen_factor=widen)

    elif name_parts[0] == "ss":
        model = ShakeNet(
            dict(
                depth=int(name_parts[1]),
                base_channels=int(name_parts[2]),
                shake_forward=True,
                shake_backward=True,
                shake_image=True,
                input_shape=(1, 3, 32, 32),
                n_classes=num_classes,
            )
        )
    elif name_parts[0] == "resnet":
        model = ResNet(num_classes=num_classes, depth=int(name_parts[1]))
    else:
        raise ValueError(f"Could not parse model name {name!r}")

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model


def load_state_dict(checkpoint_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load a PyTorch checkpoint and return a state_dict.

    Supports raw state_dict checkpoints, dict checkpoints containing a "state_dict" key,
    and DataParallel checkpoints with "module." prefixes.
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state_dict-like checkpoint at {checkpoint_path!r}, got {type(state_dict)}"
        )

    prefix = "module."
    if any(k.startswith(prefix) for k in state_dict.keys()):
        state_dict = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }

    return state_dict


def checkpoint_uses_normalize_input(state_dict: Dict[str, Any]) -> bool:
    """Infer whether a checkpoint was saved from Sequential(NormalizeInput(), model)."""

    return (
        "0.mean" in state_dict
        and "0.std" in state_dict
        and any(k.startswith("1.") for k in state_dict)
    )


def load_model_from_checkpoint(
    name: str,
    checkpoint_path: str,
    *,
    map_location: str = "cpu",
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Module:
    """Rebuild a model with the right normalization wrapper and load its weights."""

    state_dict = load_state_dict(checkpoint_path, map_location=map_location)

    # Older project WideResNet checkpoints carried an unused `sub_block1` module.
    # The refactored model removed that registration, so ignore those stale weights
    # while keeping strict loading for every other parameter.
    if name.startswith("wrn-"):
        legacy_prefixes = ("sub_block1.", "1.sub_block1.")
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith(legacy_prefixes)
        }

    model = get_model(
        name,
        num_classes=num_classes,
        normalize_input=checkpoint_uses_normalize_input(state_dict),
    )
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    return model


def forward_with_features(
    model: Module,
    x: torch.Tensor,
    *,
    require_prelogit: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return `(logits, features)` using prelogits when the model exposes them."""

    base_model: Module = model.module if isinstance(model, torch.nn.DataParallel) else model

    if (
        isinstance(base_model, Sequential)
        and len(base_model) >= 2
        and isinstance(base_model[0], NormalizeInput)
    ):
        x = base_model[0](x)
        base_model = base_model[1]

    try:
        result = base_model(x, return_prelogit=True)
    except TypeError as exc:
        logits = base_model(x)
        if require_prelogit:
            raise ValueError(
                f"Model {type(base_model).__name__} does not expose prelogit features required "
                "for LCS selection."
            ) from exc
        return logits, logits

    if isinstance(result, tuple) and len(result) == 2:
        logits, features = result
        return logits, features

    if require_prelogit:
        raise ValueError(
            f"Model {type(base_model).__name__} does not expose prelogit features required "
            "for LCS selection."
        )

    return result, result


class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def load_tinyimage_subset(other_data_path,
                          version_string='v7'):
    image_data_filename = 'tinyimage_subset_data'
    if version_string != '':
        image_data_filename += '_' + version_string
    image_data_filename += '.pickle'
    image_data_filepath = os.path.abspath(os.path.join(other_data_path, image_data_filename))
    indices_filename = 'tinyimage_subset_indices'
    if version_string != '':
        indices_filename += '_' + version_string
    indices_filename += '.json'
    indices_filepath = os.path.abspath(os.path.join(other_data_path, indices_filename))
    print('Loading indices from file {}'.format(indices_filepath))
    assert pathlib.Path(indices_filepath).is_file()
    print('Loading image data from file {}'.format(image_data_filepath))
    assert pathlib.Path(image_data_filepath).is_file()
    with open(indices_filepath, 'r') as f:
        indices = json.load(f)
    with open(image_data_filepath, 'rb') as f:
        image_data = pickle.load(f)
    num_entries = 0
    for kw, kw_indices in indices.items():
        for entry in kw_indices:
            assert entry['tinyimage_index'] in image_data
            num_entries += 1
    assert num_entries == len(image_data)
    return indices, image_data


def load_cifar10_by_keyword(unique_keywords=True, version_string='v7'):
    cifar10_keywords = load_cifar10_keywords(unique_keywords=unique_keywords,
                                             lists_for_unique=True,
                                             version_string=version_string)
    cifar10_by_keyword = {}
    for ii, keyword_entries in enumerate(cifar10_keywords):
        for entry in keyword_entries:
            cur_keyword = entry['nn_keyword']
            if cur_keyword not in cifar10_by_keyword:
                cifar10_by_keyword[cur_keyword] = []
            cifar10_by_keyword[cur_keyword].append(ii)
    return cifar10_by_keyword


def load_cifar10_keywords(other_data_path,
                          unique_keywords=True,
                          lists_for_unique=False,
                          version_string='v7'):
    filename = 'cifar10_keywords'
    if unique_keywords:
        filename += '_unique'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    keywords_filepath = os.path.abspath(os.path.join(other_data_path, filename))
    print('Loading keywords from file {}'.format(keywords_filepath))
    assert pathlib.Path(keywords_filepath).is_file()
    with open(keywords_filepath, 'r') as f:
        cifar10_keywords = json.load(f)
    if unique_keywords and lists_for_unique:
        result = []
        for entry in cifar10_keywords:
            result.append([entry])
    else:
        result = cifar10_keywords
    assert len(result) == 60000
    return result


def load_distances_to_cifar10(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    filename = 'tinyimage_cifar10_distances'
    if version_string != '':
        filename += '_' + version_string
    filename += '.json'
    filepath = os.path.abspath(os.path.join(data_path, filename))
    print('Loading distances from file {}'.format(filepath))
    assert pathlib.Path(filepath).is_file()
    with open(filepath, 'r') as f:
        tmp = json.load(f)
    if version_string == 'v4':
        assert len(tmp) == 372131
    elif version_string == 'v6':
        assert len(tmp) == 1646248
    elif version_string == 'v7':
        assert len(tmp) == 589711
    result = {}
    for k, v in tmp.items():
        result[int(k)] = v
    return result


def load_new_test_data_indices(version_string='v7'):
    data_path = os.path.join(os.path.dirname(__file__), 'data')
    filename = 'cifar10.1'
    if version_string == '':
        version_string = 'v7'
    if version_string in ['v4', 'v6', 'v7']:
        filename += '_' + version_string
    else:
        raise ValueError('Unknown dataset version "{}".'.format(version_string))
    ti_indices_data_path = os.path.join(os.path.dirname(__file__), 'data')
    ti_indices_filename = 'cifar10.1_' + version_string + '_ti_indices.json'
    ti_indices_filepath = os.path.abspath(os.path.join(ti_indices_data_path, ti_indices_filename))
    print('Loading Tiny Image indices from file {}'.format(ti_indices_filepath))
    assert pathlib.Path(ti_indices_filepath).is_file()
    with open(ti_indices_filepath, 'r') as f:
        tinyimage_indices = json.load(f)
    assert type(tinyimage_indices) is list
    if version_string == 'v6' or version_string == 'v7':
        assert len(tinyimage_indices) == 2000
    elif version_string == 'v4':
        assert len(tinyimage_indices) == 2021
    return tinyimage_indices
