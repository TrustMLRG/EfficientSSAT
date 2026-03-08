# EfficientSSAT

Official PyTorch implementation of *Improving the Efficiency of Self-Supervised Adversarial Training through Latent Clustering-Based Selection*.

Paper: [arXiv PDF](https://arxiv.org/pdf/2501.10466v1)

## Overview

EfficientSSAT studies how to improve the efficiency of self-supervised adversarial training by selecting informative auxiliary samples near the decision boundary. The repository contains the training, evaluation, and diffusion-based data generation code used for the paper.

## Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Links for Extra Data

Below are links to files containing unlabeled data from *Unlabeled Data Improves Adversarial Robustness* ([semisup-adv](https://github.com/yaircarmon/semisup-adv)):

- [500K unlabeled TinyImages with pseudo-labels](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi)

Below are links to files containing generated data from *Improving Robustness using Generated Data* ([adversarial_robustness](https://github.com/google-deepmind/deepmind-research/tree/master/adversarial_robustness)):

- [CIFAR-10 Generated Data](https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz)
- [SVHN Generated Data](https://storage.googleapis.com/dm-adversarial-robustness/svhn_ddpm.npz)

## Boundary-Focused Data Generation

`diffusionm.py` can be used to generate data near the decision boundary, and the resulting data can be used to train the final model. `diffusionmgmm.py` provides the corresponding GMM-based variant.

## Usage

Show training options:

```bash
python trainnew.py --help
```

Train with CIFAR-10:

```bash
python trainnew.py --dataset cifar10 --data_dir ./data --model wrn-28-10 --download
```

Train with auxiliary data selection:

```bash
python trainnew.py --dataset cifar10 --data_dir ./data --aux_data_filename /path/to/aux_data.npz --selection_method lcs-km --selection_model_ckpt /path/to/selector.pt --download
```

Evaluate with PGD:

```bash
python pgd_attack.py --model_path /path/to/checkpoint.pt --download
```

Evaluate with AutoAttack:

```bash
python auto_attack_cifar10.py --model_path /path/to/checkpoint.pt --download
```

Generate diffusion-based data:

```bash
python diffusionm.py --help
python diffusionmgmm.py --help
```

## Citation

```bibtex
@article{ghosh2025improving,
  title={Improving the Efficiency of Self-Supervised Adversarial Training through Latent Clustering-Based Selection},
  author={Ghosh, Somrita and Xu, Yuelin and Zhang, Xiao},
  journal={arXiv e-prints},
  pages={arXiv--2501},
  year={2025}
}
```

## Acknowledgments

The code in this repository builds on prior code and ideas from [TRADES](https://github.com/yaodongyu/TRADES) and [semisup-adv](https://github.com/yaircarmon/semisup-adv).

## License

Released under the MIT License. See `LICENSE` for details.
