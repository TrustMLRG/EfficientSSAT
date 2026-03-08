"""
Datasets with unlabeled (or pseudo-labeled) data.
"""

import logging
import os
import pickle
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.datasets import CIFAR10, SVHN

from utils import forward_with_features, load_model_from_checkpoint

DATASETS = ["cifar10", "svhn"]


def _load_aux_from_file(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if path.endswith((".pkl", ".pickle")):
        with open(path, "rb") as f:
            aux = pickle.load(f)
        if not isinstance(aux, dict) or "data" not in aux:
            raise ValueError(
                "Pickle aux file must be a dict containing a 'data' key (and optionally targets)."
            )
        targets = aux.get("extrapolated_targets", aux.get("targets"))
        return np.asarray(aux["data"]), None if targets is None else np.asarray(targets)

    if path.endswith(".npz"):
        npz = np.load(path)
        image_key = next((k for k in ("image", "images", "data") if k in npz), None)
        label_key = next((k for k in ("label", "labels", "targets") if k in npz), None)
        if image_key is None:
            raise ValueError("NPZ aux file must contain an 'image' (or 'images'/'data') array.")
        images = np.asarray(npz[image_key])
        labels = None if label_key is None else np.asarray(npz[label_key])
        return images, labels

    raise ValueError(f"Unsupported aux file type: {path!r}")


def _load_images_from_dir(directory: str) -> np.ndarray:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    )
    if not files:
        raise ValueError(f"No images found in directory: {directory!r}")

    images = []
    for fp in files:
        img = Image.open(fp).convert("RGB")
        images.append(np.array(img))
    return np.stack(images, axis=0)


def _to_uint8_images(images: np.ndarray) -> np.ndarray:
    images = np.asarray(images)
    if images.dtype == np.uint8:
        return images
    images_f = images.astype(np.float32)
    if images_f.size > 0 and images_f.max() <= 1.0:
        images_f = images_f * 255.0
    images_f = np.clip(images_f, 0.0, 255.0)
    return images_f.round().astype(np.uint8)


def _match_base_layout(images: np.ndarray, base_is_hwc: bool) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None, ...]
    if images.ndim != 4:
        raise ValueError(f"Expected aux images with 4 dims, got shape={images.shape}")

    if base_is_hwc:
        if images.shape[-1] == 3:
            return images
        if images.shape[1] == 3:
            return np.transpose(images, (0, 2, 3, 1))
    else:
        if images.shape[1] == 3:
            return images
        if images.shape[-1] == 3:
            return np.transpose(images, (0, 3, 1, 2))

    raise ValueError(
        f"Could not infer channel layout for aux images with shape={images.shape}"
    )


def _has_missing_aux_targets(aux_targets: Optional[np.ndarray]) -> bool:
    return aux_targets is None or bool(np.any(np.asarray(aux_targets) < 0))


def _fill_missing_aux_targets(
    aux_targets: Optional[np.ndarray], predicted_targets: np.ndarray
) -> np.ndarray:
    predicted_targets = np.asarray(predicted_targets, dtype=np.int64)
    if aux_targets is None:
        return predicted_targets.copy()

    aux_targets = np.asarray(aux_targets, dtype=np.int64).copy()
    missing_mask = aux_targets < 0
    if np.any(missing_mask):
        aux_targets[missing_mask] = predicted_targets[missing_mask]
    return aux_targets


def _lcs_km_order(
    feats: np.ndarray,
    kmeans_k: int,
    selection_seed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return indices sorted by LCS-KM margin (ascending).

    When fewer than 2 clusters are available (k < 2), the margin is undefined.
    In that case, fall back to a deterministic random permutation to avoid
    crashing training.
    """

    k = min(int(kmeans_k), len(feats))
    if k <= 0:
        raise ValueError("kmeans_k must be positive")
    if k < 2:
        logging.warning(
            "lcs-km requires at least 2 clusters; falling back to random selection (k=%d, n=%d).",
            k,
            len(feats),
        )
        return rng.permutation(len(feats))

    km = KMeans(n_clusters=k, random_state=selection_seed)
    km.fit(feats)
    centroids = km.cluster_centers_
    dists = np.linalg.norm(feats[:, None, :] - centroids[None, :, :], axis=2)
    part = np.partition(dists, 1, axis=1)
    margin = part[:, 1] - part[:, 0]
    return np.argsort(margin)


class _AuxSelectionDataset(Dataset):
    def __init__(self, data_np: np.ndarray, base_is_hwc: bool):
        self.data_np = data_np
        self.base_is_hwc = base_is_hwc

    def __len__(self):
        return len(self.data_np)

    def __getitem__(self, idx):
        x = self.data_np[idx]
        if self.base_is_hwc:
            x = torch.from_numpy(x).permute(2, 0, 1)
        else:
            x = torch.from_numpy(x)
        return x.float().div(255.0)


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 generated_images_dir=None,
                 selection_method="none",
                 selection_model_arch="wrn-28-10",
                 selection_model_ckpt=None,
                 n_boundary=20000,
                 n_random=30000,
                 selection_seed=42,
                 kmeans_k=10,
                 gmm_k=10,
                 selection_batch_size=256,
                 selection_num_workers=0,
                 selection_device=None,
                 train=False,
                 **kwargs):
        """A dataset with optional auxiliary data."""

        if base_dataset == "cifar10":
            self.dataset = CIFAR10(train=train, **kwargs)
        elif base_dataset == "svhn":
            split = "train" if train else "test"
            self.dataset = SVHN(split=split, **kwargs)
            if hasattr(self.dataset, "labels"):
                self.dataset.targets = self.dataset.labels
        else:
            raise ValueError(f"Dataset {base_dataset!r} not supported")

        self.base_dataset = base_dataset
        self.train = train

        # Ensure targets are list-like for extend()
        if not isinstance(self.dataset.targets, list):
            self.dataset.targets = list(self.dataset.targets)

        self.sup_indices = list(range(len(self.targets)))
        self.unsup_indices = []
        has_aux_inputs = (
            (add_svhn_extra and base_dataset == "svhn")
            or aux_data_filename is not None
            or generated_images_dir is not None
        )

        if not self.train:
            return

        if take_amount is not None:
            take_amount = int(take_amount)
            if take_amount <= 0:
                raise ValueError("take_amount must be positive")
            take_amount = min(take_amount, len(self.sup_indices))

            rng_state = np.random.get_state()
            np.random.seed(take_amount_seed)
            take_inds = np.random.choice(len(self.sup_indices), take_amount, replace=False)
            np.random.set_state(rng_state)

            data_np = np.asarray(self.data)
            targets_np = np.asarray(self.targets)

            self.data = data_np[take_inds]
            self.targets = targets_np[take_inds].tolist()
            self.sup_indices = list(range(len(self.targets)))

        if not has_aux_inputs:
            return

        base_is_hwc = (np.asarray(self.data).ndim == 4) and (np.asarray(self.data).shape[-1] == 3)

        aux_images = []
        aux_labels = []
        preserve_labels_when_unlabeled = []

        if aux_data_filename is not None:
            aux_path = str(aux_data_filename)
            if "root" in kwargs and kwargs.get("root") and not os.path.isabs(aux_path):
                candidate = os.path.join(str(kwargs["root"]), aux_path)
                if not os.path.exists(aux_path) and os.path.exists(candidate):
                    aux_path = candidate
            images_np, labels_np = _load_aux_from_file(aux_path)
            images_np = _to_uint8_images(_match_base_layout(images_np, base_is_hwc=base_is_hwc))
            aux_images.append(images_np)
            aux_labels.append(labels_np)
            preserve_labels_when_unlabeled.append(False)

        if generated_images_dir is not None:
            images_dir = str(generated_images_dir)
            if "root" in kwargs and kwargs.get("root") and not os.path.isabs(images_dir):
                candidate = os.path.join(str(kwargs["root"]), images_dir)
                if not os.path.exists(images_dir) and os.path.exists(candidate):
                    images_dir = candidate
            images_np = _to_uint8_images(
                _match_base_layout(_load_images_from_dir(images_dir), base_is_hwc=base_is_hwc)
            )
            aux_images.append(images_np)
            aux_labels.append(None)
            preserve_labels_when_unlabeled.append(False)

        if add_svhn_extra and base_dataset == "svhn":
            svhn_extra = SVHN(split="extra", **kwargs)
            extra_labels = getattr(svhn_extra, "labels", getattr(svhn_extra, "targets", None))
            extra_data = _to_uint8_images(
                _match_base_layout(np.asarray(svhn_extra.data), base_is_hwc=base_is_hwc)
            )
            aux_images.append(extra_data)
            aux_labels.append(None if extra_labels is None else np.asarray(extra_labels))
            preserve_labels_when_unlabeled.append(True)

        aux_data = np.concatenate([np.asarray(x) for x in aux_images], axis=0)
        aux_targets = None
        aux_targets_when_unlabeled = None
        if any(lbl is not None for lbl in aux_labels):
            labels_concat = []
            preserved_concat = []
            for lbl, img, keep_labels in zip(
                aux_labels, aux_images, preserve_labels_when_unlabeled
            ):
                if lbl is None:
                    labels = np.full((len(img),), -1, dtype=np.int64)
                else:
                    labels = np.asarray(lbl).reshape(-1)
                labels_concat.append(labels)
                if keep_labels:
                    preserved_concat.append(labels)
                else:
                    preserved_concat.append(np.full((len(img),), -1, dtype=np.int64))
            aux_targets = np.concatenate(labels_concat, axis=0)
            aux_targets_when_unlabeled = np.concatenate(preserved_concat, axis=0)

        aux_data = _to_uint8_images(aux_data)

        if aux_targets is not None and len(aux_targets) != len(aux_data):
            raise ValueError(
                f"Aux targets length ({len(aux_targets)}) does not match aux data length ({len(aux_data)})"
            )

        if aux_take_amount is not None:
            aux_take_amount = int(aux_take_amount)
            if aux_take_amount <= 0:
                raise ValueError("aux_take_amount must be positive")
            aux_take_amount = min(aux_take_amount, len(aux_data))
            rng_state = np.random.get_state()
            np.random.seed(selection_seed)
            keep = np.random.choice(len(aux_data), aux_take_amount, replace=False)
            np.random.set_state(rng_state)
            aux_data = aux_data[keep]
            if aux_targets is not None:
                aux_targets = aux_targets[keep]
            if aux_targets_when_unlabeled is not None:
                aux_targets_when_unlabeled = aux_targets_when_unlabeled[keep]

        selection_method = (selection_method or "none").lower()
        if selection_method not in {"none", "predconf", "lcs-km", "lcs-gmm"}:
            raise ValueError(f"Unknown selection_method: {selection_method!r}")

        need_model = selection_method != "none" or (add_aux_labels and _has_missing_aux_targets(aux_targets))
        features_np = None
        conf_np = None
        yhat_np = None
        need_prelogits = selection_method in {"lcs-km", "lcs-gmm"}
        need_confidence = selection_method == "predconf"
        need_predictions = add_aux_labels and _has_missing_aux_targets(aux_targets)

        if need_model:
            if selection_model_ckpt is None:
                raise ValueError(
                    "selection_model_ckpt is required for selection_method != 'none' "
                    "or when pseudo-labels are needed."
                )

            device = (
                selection_device
                if isinstance(selection_device, torch.device)
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            model = load_model_from_checkpoint(
                selection_model_arch,
                selection_model_ckpt,
                map_location="cpu",
                num_classes=10,
                device=device,
            )
            model.eval()

            loader = DataLoader(
                _AuxSelectionDataset(aux_data, base_is_hwc=base_is_hwc),
                batch_size=selection_batch_size,
                shuffle=False,
                num_workers=selection_num_workers,
            )

            features_list = [] if need_prelogits else None
            conf_list = [] if need_confidence else None
            yhat_list = [] if need_predictions else None
            with torch.no_grad():
                for xb in loader:
                    xb = xb.to(device)
                    logits, features = forward_with_features(
                        model, xb, require_prelogit=need_prelogits
                    )
                    if need_prelogits:
                        features_list.append(features.detach().cpu())
                    if need_confidence:
                        probs = F.softmax(logits, dim=1)
                        conf_list.append(probs.max(dim=1).values.detach().cpu())
                    if need_predictions:
                        yhat_list.append(logits.argmax(dim=1).detach().cpu())

            if need_prelogits:
                features_np = torch.cat(features_list, dim=0).numpy()
            if need_confidence:
                conf_np = torch.cat(conf_list, dim=0).numpy()
            if need_predictions:
                yhat_np = torch.cat(yhat_list, dim=0).numpy()

        if add_aux_labels:
            if _has_missing_aux_targets(aux_targets):
                aux_targets = _fill_missing_aux_targets(aux_targets, yhat_np)
            else:
                aux_targets = np.asarray(aux_targets, dtype=np.int64)
        else:
            if aux_targets_when_unlabeled is None:
                aux_targets = np.full((len(aux_data),), -1, dtype=np.int64)
            else:
                aux_targets = np.asarray(aux_targets_when_unlabeled, dtype=np.int64)

        rng = np.random.default_rng(selection_seed)
        if selection_method == "none":
            final_inds = np.arange(len(aux_data))
        else:
            selected = np.arange(len(aux_data))

            if selection_method == "predconf":
                order = np.argsort(conf_np)
                selected = order
            elif selection_method == "lcs-km":
                selected = _lcs_km_order(
                    feats=features_np,
                    kmeans_k=kmeans_k,
                    selection_seed=selection_seed,
                    rng=rng,
                )
            elif selection_method == "lcs-gmm":
                feats = features_np
                k = min(int(gmm_k), len(feats))
                if k <= 0:
                    raise ValueError("gmm_k must be positive")
                if len(feats) < 2 or k < 2:
                    logging.warning(
                        "lcs-gmm requires at least 2 samples and clusters; falling back to random selection (k=%d, n=%d).",
                        k,
                        len(feats),
                    )
                    selected = rng.permutation(len(feats))
                else:
                    gmm = GaussianMixture(n_components=k, random_state=selection_seed)
                    gmm.fit(feats)
                    nll = -gmm.score_samples(feats)
                    selected = np.argsort(nll)[::-1]

            n_boundary = min(int(n_boundary), len(selected))
            boundary_inds = selected[:n_boundary]
            remaining = np.setdiff1d(np.arange(len(aux_data)), boundary_inds, assume_unique=False)
            n_random = min(int(n_random), len(remaining))
            if n_random > 0:
                rand_inds = rng.choice(remaining, size=n_random, replace=False)
                final_inds = np.concatenate([boundary_inds, rand_inds])
            else:
                final_inds = boundary_inds

        selected_data = aux_data[final_inds]
        selected_targets = aux_targets[final_inds].tolist()

        orig_len = len(self.data)
        self.data = np.concatenate([np.asarray(self.data), selected_data], axis=0)
        self.targets = list(self.targets) + selected_targets
        self.unsup_indices = list(range(orig_len, orig_len + len(selected_data)))



    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets  # because torchvision is annoying
        return self.dataset[item]

    def __repr__(self):
        fmt_str = 'Semisupervised Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Training: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.dataset.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.dataset.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class SemiSupervisedSampler(Sampler):
    """Balanced sampling from the labeled and unlabeled data"""
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5,
                 num_batches=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            self.unsup_inds = []
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        if len(self.unsup_inds) == 0:
            unsup_fraction = 0.0

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(
                np.ceil(len(self.sup_inds) / self.sup_batch_size))

        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in
                                  torch.randint(high=len(self.unsup_inds),
                                                size=(
                                                    self.batch_size - len(
                                                        batch),),
                                                dtype=torch.int64)])
                # Shuffle mixed batches before yielding them to avoid
                # unstable batch-normalization behavior under DataParallel.
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
