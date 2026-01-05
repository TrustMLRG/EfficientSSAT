
# Improving the Efficiency of Self-Supervised Adversarial Training through Latent Clustering-Based Selection
  
This repository contains code for Improving the Efficiency of Self-Supervised Adversarial Training through Latent Clustering-Based Selection

## Links for 'extra' data 

Below are links to files containing unlabeled data from the paper Unlabeled data improves adversarial robustness(https://github.com/yaircarmon/semisup-adv):

- [500K unlabeled data from TinyImages (with pseudo-labels)](https://drive.google.com/open?id=1LTw3Sb5QoiCCN-6Y5PEKkq9C9W60w-Hi)

Below are links to files containing generated data from the paper Improving Robustness using Generated Data(https://github.com/google-deepmind/deepmindresearch/tree/master/adversarial_robustness)

- [CIFAR-10 Generated Data (https://storage.googleapis.com/dm-adversarial-robustness/cifar10_ddpm.npz)]
- [SVHN Generated Data (https://storage.googleapis.com/dm-adversarial-robustness/svhn_ddpm.npz)]

## In order to directly generate data near the boundary 
- diffusionm.py can be used to generate data near the boundary and this data can be used to train the final model.


The code in this repo is based on code from the following sources:  
- TRADES: https://github.com/yaodongyu/TRADES  
- Unlabeled Data Improves Adversarial Robustness:  https://github.com/yaircarmon/semisup-adv

## Running SSAT:
To run our code you first choose the original labelled dataset, specify the amount of extra data you want and give the path to the appropiate generated or unlabeled data file.
