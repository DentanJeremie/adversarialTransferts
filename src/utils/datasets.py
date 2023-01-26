import torch
import torchvision
import numpy as np
import time
import imageio
import os

from zipfile import ZipFile

from src.utils.pathtools import project
from src.utils.logging import logger

# --------------- CIFAR100 ---------------

BATCH_SIZE = 1
NAME = 'CIFAR100'

class Datasets_CIFAR100():

    def __init__(self):
        self._dataset: torchvision.datasets.VisionDataset = None
        self._loader: torch.utils.data.dataloader.DataLoader = None

    @property
    def dataset(self):
        if self._dataset is None:
            logger.info(f'Downloading {NAME} dataset...')
            self._dataset = torchvision.datasets.CIFAR100(
                root=project.data,
                download = True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
            )
        return self._dataset

    @property
    def loader(self):
        if self._loader is None:
            logger.info(f'Building dataloader...')
            self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        return self._loader


cifar100 = Datasets_CIFAR100()

# --------------- Tiny_imagenet ---------------

class Datasets_tiny_imagenet():

    def __init__(self):
        self._dataset: torchvision.datasets.VisionDataset = None
        self._loader: torch.utils.data.dataloader.DataLoader = None

    # Processing
   
    @property
    def dataset(self):
        if not os.path.isdir('../../data/tiny-imagenet-200'):
            with ZipFile('../../data/tiny-imagenet-200.zip') as zf:
                zf.extractall('../../data')
        
        id_dict = {}
        for i, line in enumerate(open( '../../data/tiny-imagenet-200/wnids.txt', 'r')):
            id_dict[line.replace('\n', '')] = i
        
        print('loading dataset')
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()

        # Decomment here to consider the whole dataset

        """for key, value in id_dict.items():
            train_data += [imageio.imread('../../data/tiny-imagenet-200/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), pilmode='RGB') for i in range(450)]
            train_labels_ = np.array([[0]*200]*450)
            train_labels_[:, value] = 1
            train_labels += train_labels_.tolist()"""

        for line in open( '../../data/tiny-imagenet-200/val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(imageio.imread('../../data/tiny-imagenet-200/val/images/{}'.format(img_name) ,pilmode='RGB'))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()

        print('finished loading dataset, in {} seconds'.format(time.time() - t))
        train_data, train_labels, test_data, test_labels = np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

        tensor_x = torch.Tensor(test_data) #torch.cat((torch.Tensor(train_data),torch.Tensor(test_data))) # transform to torch tensor
        tensor_y = torch.Tensor(test_labels) #torch.cat((torch.Tensor(train_labels),torch.Tensor(test_labels)))

        tensor_x = torch.permute(tensor_x, (0, 3, 1, 2))/255
        self._dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your dataset

        return self._dataset

    @property
    def loader(self):
        if self._loader is None:
            logger.info(f'Building dataloader...')
            self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        return self._loader

    # Processing


tiny_imagenet = Datasets_tiny_imagenet()