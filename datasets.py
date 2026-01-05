"""
Datasets with unlabeled (or pseudo-labeled) data
"""

from torchvision.datasets import CIFAR10, SVHN
from torch.utils.data import Sampler, Dataset
from utils import get_model
from utils import extract_penultimate_features
import torch
import numpy as np
import torch.nn as nn
import random
import contextlib
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from models.wideresnet import *
from models.resnet import *
#from trades import trades_loss
from sklearn.mixture import GaussianMixture




import pdb

import os
import pickle

import logging

DATASETS = ['cifar10', 'svhn']


class SemiSupervisedDataset(Dataset):
    def __init__(self,
                 base_dataset='cifar10',
                 take_amount=None,
                 take_amount_seed=13,
                 add_svhn_extra=False,
                 aux_data_filename=None,
                 add_aux_labels=False,
                 aux_take_amount=None,
                 train=False,
                 **kwargs):
        """A dataset with auxiliary pseudo-labeled data"""

        if base_dataset == 'cifar10':
            self.dataset = CIFAR10(train=train, **kwargs)
        elif base_dataset == 'svhn':
            if train:
                self.dataset = SVHN(split='train', **kwargs)
            else:
                self.dataset = SVHN(split='test', **kwargs)
            # because torchvision is annoying
            self.dataset.targets = self.dataset.labels
            self.targets = list(self.targets)

            if train and add_svhn_extra:
                npzfile = np.load('/TRADES-master/svhn_20percent_60:40/svhn11_ddpm.npz')
                images = npzfile['image']
                labels = npzfile['label']
                # Set a random seed for reproducibility (optional)
                np.random.seed(42)
                self.unsup_indices = []
                orig_len = len(self.data)
                self.sup_indices = list(range(len(self.targets)))
                method= "lcs-km"
                checkpoint = torch.load('/TRADES-master/svhn_test16_8/model-wideres-epoch75.pt')
                model = WideResNet()
                model = nn.DataParallel(model).cuda()
                model.load_state_dict(checkpoint)
                model.eval()
                numboundary = 31800
                numrandom = 21200
                predictions = []
                index_confidence_pairs = []
                latent_representations = []
                with torch.no_grad():
                    for index in range(len(images)):
                        data = images[index]
                        data = torch.tensor(data).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32).cuda()
                        output = model(data)
                        predicted_classes = torch.argmax(output, dim=1)
                        prediction_confidences = torch.softmax(output, dim=1).max(dim=1).values
                        index_confidence_pairs.append((index, prediction_confidences.item()))
                        predictions.append(predicted_classes)
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        latent_rep = extract_penultimate_features(model, data, device)
                        #latent_rep = latent_rep.reshape(latent_representations.shape[0], -1)
                        latent_representations.append(latent_rep)
                latent_representations = np.vstack(latent_representations)
                latent_representations = latent_representations.reshape(latent_representations.shape[0], -1)

                # Perform extrapolation based on the selected method
                if method == 'predconf':
                    sorted_index_confidence_pairs = sorted(index_confidence_pairs, key=lambda x: x[1])
                    sorted_indices = [pair[0] for pair in sorted_index_confidence_pairs]
                    remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                    random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                    low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                elif method == 'lcs-km':
                    num_prototypes = 10
                    kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
                    kmeans.fit(latent_representations)
                    centroids = kmeans.cluster_centers_
                    distances = np.linalg.norm(latent_representations[:, np.newaxis] - centroids, axis=2)
                    sorted_indices = np.argsort(np.partition(distances, 1)[:, 1] - distances.min(axis=1))
                    remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                    random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                    low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                elif method == 'lcs-gmm':
                    from sklearn.mixture import GaussianMixture
                    num_components = 10
                    gmm = GaussianMixture(n_components=num_components, random_state=42)
                    gmm.fit(latent_representations)
                    distances = -gmm.score_samples(latent_representations)  # Negative log likelihood
                    sorted_indices = np.argsort(distances)
                    remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                    random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                    low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                else:
                    raise ValueError(f"Unknown method: {method}")

                # Updating the dataset
                new_extrapolated_targets = torch.cat(predictions, dim=0)
                new_targets_list = new_extrapolated_targets.tolist()
                selected_data = images[low_confidence_indices]
                selected_targets = labels[low_confidence_indices]

                self.targets = np.concatenate([self.targets, selected_targets])
                selected_data = torch.from_numpy(selected_data).permute(0, 3, 1, 2)
                self.data = np.concatenate([self.data, selected_data])
                self.unsup_indices.extend(range(orig_len, orig_len + len(selected_targets)))

        else:
            raise ValueError('Dataset %s not supported' % base_dataset)
        self.base_dataset = base_dataset
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices),
                                             take_amount, replace=False)
                np.random.set_state(rng_state)

                logger = logging.getLogger()
                logger.info('Randomly taking only %d/%d examples from training'
                            ' set, seed=%d, indices=%s',
                            take_amount, len(self.sup_indices),
                            take_amount_seed, take_inds)
                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            #self.sup_indices = list(range(len(self.targets)))
            #self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = aux_data_filename
                print("Loading data from %s" % aux_path)
                npzfile = np.load('semisup_new/semisup-adv-master/cifar10_ddpm.npz')
                images = npzfile['image']
                labels = npzfile['label']
                np.random.seed(42)
                self.unsup_indices = []
                method= "lcs-km"
                # Randomly select 200 indices from the total number of images
                random_indices = np.random.choice(len(images), 500, replace=False)

                # Use these indices to select the corresponding images and labels
                images = images[random_indices]

                labels = labels[random_indices]
                with open(aux_path, 'rb') as f:
                    aux = pickle.load(f)
                aux_data = aux['data']
                aux_targets = aux['extrapolated_targets']
                orig_len = len(self.data)
                self.sup_indices = list(range(len(self.targets)))
                #selected_indices = random.sample(range(total_indices), num_indices_to_select)
                #aux_data = aux_data[selected_indices]
                #aux_targets =aux_targets[selected_indices]
                #self.data = np.concatenate((self.data, aux_data), axis=0)
                if aux_take_amount is not None:
                    #rng_state = np.random.get_state()
                    #np.random.seed(take_amount_seed)
                    #take_inds = np.random.choice(len(aux_data),
                                                 #aux_take_amount, replace=False)
                    #np.random.set_state(rng_state)

                    #logger = logging.getLogger()
                    #logger.info(
                        #'Randomly taking only %d/%d examples from aux data'
                        #' set, seed=%d, indices=%s',
                        #aux_take_amount, len(aux_data),
                       # take_amount_seed, take_inds)
                    checkpoint1 = torch.load('/trades/TRADES-master/cifar10std/model-wideres-epoch50.pt')
                    num_classes = 10
                
                    #normalize_input = checkpoint.get('normalize_input', False)
                    model1 = 'wrn-28-10'
                    model = get_model(model1, num_classes=num_classes,
                                        normalize_input=False)
                    model = nn.DataParallel(model).cuda()
                    model.load_state_dict(checkpoint1) 
                    model.eval()
                    numboundary = 30000
                    numrandom = 20000
                    #with torch.no_grad():
                        #svhn_extra.labels = model(svhn_extra.data)
                    #input_data=[]
                    predictions= []
                    entropies=[]
                    low_confidence_indices=[]
                    step_size=0.001
                    epsilon=0.031
                    perturb_steps=1
                    beta=6.0
                    #total_indices = 531130
                    #num_indices_to_select = 2838
                    #aux_data = torch.from_numpy(aux_data)
                    #selected_indices = random.sample(range(total_indices), num_indices_to_select)
                    transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])
                    entropies=[]
                    gradients_magnitude = []
                    num_prototypes = 10  # Adjust as needed
                    num_samples = aux_data.shape[0]
                    flattened_data = aux_data.reshape(num_samples, -1) 
                    index_confidence_pairs = []
                    # Apply K-means clustering to the data
                    #kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
                    #kmeans.fit(flattened_data)

                    # Get the centroids of the clusters
                    #centroids = kmeans.cluster_centers_

                    # Calculate distances from each data point to its nearest centroid
                    #distances = np.min(np.linalg.norm(flattened_data[:, np.newaxis] - centroids, axis=2), axis=1)

                    # Sort data points based on distances
                    #sorted_indices = np.argsort(distances)

                    # Select boundary examples
                    #low_confidence_indices = aux_data[sorted_indices[:39000]]
                    latent_representations = []
                    with torch.no_grad():
                         for index, (data, target) in enumerate(zip(aux_data, aux_targets)): 
                         #for index in range(len(images)):# Assuming svhn_extra is a DataLoader or similar iterable
                            #data =images[index]
                            data_orig=data
                            #data=torch.tensor(data)
                            #data = data.unsqueeze(0)  # Add a batch dimension

                            #data = data[np.newaxis, :]
                            #predictions.append(model(data))
                            data_type = data.shape
                            #print(f"Data at index {index} has type: {data_type}")
                            #data = data.permute(0, 3, 1, 2)
                            #data = data.to(torch.float32)  # Assuming svhn_extra is a DataLoader or similar iterable
                            data_orig=data
                            data_type = data.shape
                            #print(f"Data at index {index} has type: {data_type}")
                            data_pil = Image.fromarray(data)
                            data = transform_train(data_pil)
                            #data = data.unsqueeze(0)
                            data = data[None, :]
                            data_type = data.shape
                            #print(f"Data at index {index} has type: {data_type}")
                            #data.requires_grad = True
                            output = model(data)
                            predicted_classes = torch.argmax(output, dim=1)
                            prediction_confidences = torch.softmax(output, dim=1).max(dim=1).values
                            index_confidence_pairs.append((index, prediction_confidences.item()))
                            predictions.append(predicted_classes)
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            latent_rep = extract_penultimate_features(model, data, device)
                        #latent_rep = latent_rep.reshape(latent_representations.shape[0], -1)
                            latent_representations.append(latent_rep)
                            #latent_representations.append(output.cpu().numpy())
                            #target_tensor = torch.tensor([target], device=output.device)
                            #loss = nn.CrossEntropyLoss()(output, target_tensor)
                            #loss = nn.CrossEntropyLoss()(output, torch.tensor([target]))
                            #model.zero_grad()
                            #loss.backward()
                            # Add a batch dimension
                            #predictions.append(model(data))
                            #gradient_magnitude = data.grad.abs().max().item()
                            #gradients_magnitude.append(gradient_magnitude)
                            #output = model(data)
                            #predicted_classes = torch.argmax(output, dim=1)
                            #prediction_confidences = torch.softmax(output, dim=1).max(dim=1).values
                            #if index in selected_indices:
                            #entropy = -torch.sum(output * torch.log2(output + 1e-10), dim=1).cpu().numpy()
                            #entropies.append(entropy.item())
                            #if (prediction_confidences <= 0.7 ): # change this to 0.4 delete the predicted classes part
                                #predictions.append(predicted_classes)
                                #self.data = np.concatenate([self.data,data_orig.numpy()])
                                #input_data.append(data_orig)
                                #low_confidence_indices.append(index)
                    #high_confidence_indices = [index for index in range(len(prediction_confidences)) if prediction_confidences[index] < 0.5]
                    #low_confidence_indices = np.argsort(entropies)[-10000:]
                    #new_extrapolated_targets = torch.cat(predictions, dim=0)
                    #new_targets_list = new_extrapolated_targets.tolist()
                    # Stack the latent representations into a numpy array
                    latent_representations = np.vstack(latent_representations)
                    latent_representations = latent_representations.reshape(latent_representations.shape[0], -1)
                    #latent_representations = np.vstack(latent_representations)
                    new_extrapolated_targets = torch.cat(predictions, dim=0)
                    new_targets_list = new_extrapolated_targets.tolist()
                    # Apply K-means clustering to the latent representations
                    # Perform extrapolation based on the selected method
                    if method == 'predconf':
                        sorted_index_confidence_pairs = sorted(index_confidence_pairs, key=lambda x: x[1])
                        sorted_indices = [pair[0] for pair in sorted_index_confidence_pairs]
                        remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                        random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                        low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                    elif method == 'lcs-km':
                        num_prototypes = 10
                        kmeans = KMeans(n_clusters=num_prototypes, random_state=42)
                        kmeans.fit(latent_representations)
                        centroids = kmeans.cluster_centers_
                        distances = np.linalg.norm(latent_representations[:, np.newaxis] - centroids, axis=2)
                        sorted_indices = np.argsort(np.partition(distances, 1)[:, 1] - distances.min(axis=1))
                        remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                        random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                        low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                    elif method == 'lcs-gmm':
                        from sklearn.mixture import GaussianMixture
                        num_components = 10
                        gmm = GaussianMixture(n_components=num_components, random_state=42)
                        gmm.fit(latent_representations)
                        distances = -gmm.score_samples(latent_representations)  # Negative log likelihood
                        sorted_indices = np.argsort(distances)
                        remaining_indices = np.setdiff1d(np.arange(len(images)), sorted_indices[:numboundary])
                        random_indices = np.random.choice(remaining_indices, numrandom, replace=False)
                        low_confidence_indices = np.concatenate((sorted_indices[:numboundary], random_indices))

                    selected_targets = aux_targets[low_confidence_indices]
                    #selected_targets = aux_targets
                    selected_data = aux_data[low_confidence_indices]
                    self.data = np.concatenate([self.data, selected_data])
                    print("lenself.data",len(self.data))
                
                    #self.data = np.concatenate((self.data, aux_data), axis=0)
                if not add_aux_labels:
                    #self.targets.extend([-1] * len(aux_data))
                    # Extend with -1 for the first 19500 elements
                    self.targets.extend([-1] * min(len(aux_data), 15600))
                    # Extend with values from aux_targets for the rest
                    if len(aux_targets) > 19500:
                        # Fill the remaining elements with actual labels
                        self.targets.extend(aux_targets)
                else:
                    self.targets.extend(selected_targets)
                    print(len(self.targets))
                # note that we use unsup indices to track the labeled datapoints
                # whose labels are "fake"
                self.unsup_indices = []
                self.unsup_indices.extend(
                    range(orig_len, orig_len+len(selected_data)))
            else:
                self.unsup_indices = []
                # Use these indices to select the corresponding images and labels
                orig_len = len(self.data)
                self.sup_indices = list(range(len(self.targets)))
                # === Step 1: Define the directory containing saved images generated/extra===
                generated_dir = "/home/c01sogh/CISPA-home/trades/TRADES-master/generated_risky_samples_2/"

                # === Step 2: Define transform (same as used in CIFAR-10) ===
                transform = transforms.Compose([
                    transforms.ToTensor(),  # Converts [0, 255] PIL image to [0.0, 1.0] float tensor
                ])

                # === Step 3: Load saved PNG images ===
                image_tensors = []
                filenames = sorted([f for f in os.listdir(generated_dir) if f.endswith(".png")])
                for fname in filenames:
                    img_path = os.path.join(generated_dir, fname)
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img)
                    image_tensors.append(img_tensor)

                image_tensor_batch = torch.stack(image_tensors)  # Shape: [N, 3, 32, 32]

                # === Step 4: Load your classifier model ===
               

                checkpoint1 = torch.load('/home/c01sogh/CISPA-home/trades/TRADES-master/cifar10_20percent_dataratio0.7_123generated_DDPM_beta0.4/model-wideres-epoch100.pt')
                num_classes = 10
                
                    #normalize_input = checkpoint.get('normalize_input', False)
                model1 = 'wrn-28-10'
                model = get_model(model1, num_classes=num_classes,
                                    normalize_input=False)
                model = nn.DataParallel(model).cuda()
                model.load_state_dict(checkpoint1) 
                model.eval()
                # === Step 5: Predict labels for the generated images ===
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                predicted_labels = []

                with torch.no_grad():
                    image_tensor_batch = image_tensor_batch.to(device)
                    outputs = model(image_tensor_batch)
                    predicted_labels = torch.argmax(outputs, dim=1).cpu().tolist()

                # === Step 6: Add these to your dataset ===
                generated_data_np = image_tensor_batch.mul(255).clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()  # Convert to NumPy
                #generated_data_np = np.transpose(generated_data_np, (0, 2, 3, 1))  # Convert from [N, 3, 32, 32] to [N, 32, 32, 3]


                orig_len = len(self.data)
                self.data = np.concatenate([self.data, generated_data_np])
                self.targets.extend(predicted_labels)
                self.unsup_indices.extend(range(orig_len, orig_len + len(generated_data_np)))

                print(f"✅ {len(generated_data_np)} risky samples added to CIFAR-10 training data.")


            logger = logging.getLogger()
            logger.info("Training set")
            logger.info("Number of training samples: %d", len(self.targets))
            logger.info("Number of supervised samples: %d",
                        len(self.sup_indices))
            logger.info("Number of unsup samples: %d", len(self.unsup_indices))
            logger.info("Label (and pseudo-label) histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of training data: %s", np.shape(self.data))

        # Test set
        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            logger = logging.getLogger()
            logger.info("Test set")
            logger.info("Number of samples: %d", len(self.targets))
            logger.info("Label histogram: %s",
                        tuple(
                            zip(*np.unique(self.targets, return_counts=True))))
            logger.info("Shape of data: %s", np.shape(self.data))



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
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

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
                # this shuffle operation is very important, without it
                # batch-norm / DataParallel hell ensues
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches
