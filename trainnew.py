from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss



import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from utils import get_model


from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS



from autoaugment import CIFAR10Policy
from cutout import Cutout

import logging
parser = argparse.ArgumentParser(
    description='PyTorch TRADES Adversarial Training')

# Dataset config
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=DATASETS,
                    help='The dataset to use for training)')
parser.add_argument('--data_dir', default='data', type=str,
                    help='Directory where datasets are located')
parser.add_argument('--svhn_extra', action='store_true', default=False,
                    help='Adds the extra SVHN data')

# Model config
parser.add_argument('--model', '-m', default='ResNet18', type=str,
                    help='Name of the model (see utils.get_model)')
parser.add_argument('--model_dir', default='/home/c01sogh/CISPA-home/trades/TRADES-master/cifar10_10percent_wrnpgd-10',
                    help='Directory of model for saving checkpoint')
parser.add_argument('--overwrite', action='store_true', default=True,
                    help='Cancels the run if an appropriate checkpoint is found')
parser.add_argument('--normalize_input', action='store_true', default=False,
                    help='Apply standard CIFAR normalization first thing '
                         'in the network (as part of the model, not in the data'
                         ' fetching pipline)')

# Logging and checkpointing
parser.add_argument('--log_interval', type=int, default=5,
                    help='Number of batches between logging of training status')
parser.add_argument('--save_freq', default=25, type=int,
                    help='Checkpoint save frequency (in epochs)')

# Generic training configs
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed. '
                         'Note: fixing the random seed does not give complete '
                         'reproducibility. See '
                         'https://pytorch.org/docs/stable/notes/randomness.html')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--test_batch_size', type=int, default=500, metavar='N',
                    help='Input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='Number of epochs to train. '
                         'Note: we arbitrarily define an epoch as a pass '
                         'through 50K datapoints. This is convenient for '
                         'comparison with standard CIFAR-10 training '
                         'configurations.')

# Eval config
parser.add_argument('--eval_freq', default=1, type=int,
                    help='Eval frequency (in epochs)')
parser.add_argument('--train_eval_batches', default=None, type=int,
                    help='Maximum number for batches in training set eval')
parser.add_argument('--eval_attack_batches', default=1, type=int,
                    help='Number of eval batches to attack with PGD or certify '
                         'with randomized smoothing')

# Optimizer config
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float)
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=('trades', 'trades_fixed', 'cosine', 'wrn'),
                    help='Learning rate schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='Use extragrdient steps')

# Adversarial / stability training config
parser.add_argument('--loss', default='trades', type=str,
                    choices=('trades', 'noise'),
                    help='Which loss to use: TRADES-like KL regularization '
                         'or noise augmentation')

parser.add_argument('--distance', '-d', default='l_inf', type=str,
                    help='Metric for attack model: l_inf uses adversarial '
                         'training and l_2 uses stability training and '
                         'randomized smoothing certification',
                    choices=['l_inf', 'l_2'])
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='Adversarial perturbation size (takes the role of'
                         ' sigma for stability training)')

parser.add_argument('--pgd_num_steps', default=1, type=int,
                    help='number of pgd steps in adversarial training')
parser.add_argument('--pgd_step_size', default=0.007,
                    help='pgd steps size in adversarial training', type=float)
parser.add_argument('--beta', default=6.0, type=float,
                    help='stability regularization, i.e., 1/lambda in TRADES')

# Semi-supervised training configuration
parser.add_argument('--aux_data_filename', default= None, type=str,
                    help='Path to pickle file containing unlabeled data and '
                         'pseudo-labels used for RST')

parser.add_argument('--unsup_fraction', default=0.5, type=float,
                    help='Fraction of unlabeled examples in each batch; '
                         'implicitly sets the weight of unlabeled data in the '
                         'loss. If set to -1, batches are sampled from a '
                         'single pool')
parser.add_argument('--aux_take_amount', default=5, type=int,
                    help='Number of random aux examples to retain. '
                         'None retains all aux data.')

parser.add_argument('--remove_pseudo_labels', action='store_true',
                    default=False,
                    help='Performs training without pseudo-labels (rVAT)')
parser.add_argument('--entropy_weight', type=float,
                    default=0.0, help='Weight on entropy loss')

# Additional aggressive data augmentation
parser.add_argument('--autoaugment', action='store_true', default=False,
                    help='Use autoaugment for data augmentation')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='Use cutout for data augmentation')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')


args = parser.parse_args()

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.model_dir, 'training.log')),
        logging.StreamHandler()
    ])
logger = logging.getLogger()

logging.info('Robust self-training')
logging.info('Args: %s', args)
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
#transform_train = transforms.Compose([
    #transforms.ToTensor(),
#])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = SemiSupervisedDataset(base_dataset=args.dataset,
                                 add_svhn_extra=args.svhn_extra,
                                 root=args.data_dir, train=True,
                                 download=True, transform=transform_train,
                                 aux_data_filename=args.aux_data_filename,
                                 add_aux_labels=not args.remove_pseudo_labels,
                                 aux_take_amount=args.aux_take_amount)

# num_batches=50000 enforces the definition of an "epoch" as passing through 50K
# datapoints
# TODO: make sure that this code works also when trainset.unsup_indices=[]
train_batch_sampler = SemiSupervisedSampler(
    trainset.sup_indices, trainset.unsup_indices,
    args.batch_size, args.unsup_fraction,
    num_batches=int(np.ceil(50000 / args.batch_size)))
epoch_size = len(train_batch_sampler) * args.batch_size

kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler, **kwargs)

testset = SemiSupervisedDataset(base_dataset=args.dataset,
                                root=args.data_dir, train=False,
                                download=True,
                                transform=transform_test)
test_loader = DataLoader(testset, batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.pgd_step_size,
                           epsilon=args.epsilon,
               		   perturb_steps=10,
                	   beta=1.0,
                	   distance='l_inf')
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), epoch_size,
                       100. * batch_idx / epoch_size, loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    logging.info('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, epoch_size,
        100. * correct / epoch_size))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logging.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 75:
        lr = args.lr * 0.1
    if epoch >= 90:
        lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    #model = ResNet18().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #checkpoint1 = torch.load('/home/c01sogh/CISPA-home/trades/TRADES-master/cifar10_20percent_dataratio0.3generated_DDPM_beta100percentrandomsamples/model-wideres-epoch75.pt')
    #num_classes = 10
                
        #normalize_input = checkpoint.get('normalize_input', False)
    model = 'wrn-28-10'
    #model = get_model(model, num_classes=num_classes,
                        #normalize_input=False)
    #model.load_state_dict(checkpoint1)
    num_classes = 10
    model = get_model(model, num_classes=num_classes,
                      normalize_input=args.normalize_input)
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          nesterov=args.nesterov)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        eval_train(model, device, train_loader)
        eval_test(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-wideres-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-wideres-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()
