import time
import sys
from zipfile import ZipFile

import imageio
import numpy as np
import requests
import torch
import torchvision

from src.utils.pathtools import project
from src.utils.logging import logger

TINY_IMAGENET_DOWNLOAD = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
BATCH_SIZE = 32


# Code partly taken from https://github.com/TheAthleticCoder/Tiny-ImageNet-200

class Datasets_tiny_imagenet():

    def __init__(self):
        self._dataset: torchvision.datasets.VisionDataset = None
        self._loader: torch.utils.data.dataloader.DataLoader = None
   
    @property
    def dataset(self):
        if self._dataset is None:
            self.build_dataset()
        return self._dataset
        

    @property
    def loader(self):
        if self._loader is None:
            logger.info(f'Building dataloader...')
            self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True)
        return self._loader

# -------------------- DOWNLOAD ---------------------

    def build_dataset(self):
        """TO BE COMPLETED"""
        
        if not project.tiny_imagenet.exists() or len(list(project.tiny_imagenet.iterdir())) <= 2:
            logger.info('Tiny-Imagenet-200 dataset not found')

            if not project.tiny_imagenet_zip.exists():
                logger.info('Tiny-Imagenet-200 dataset zip not found, downloading it...')
                response = requests.get(TINY_IMAGENET_DOWNLOAD, stream=True)
                with project.tiny_imagenet_zip.open('wb') as f:
                    dl = 0
                    total_length = response.headers.get('content-length')
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\Progression: [%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()

                sys.stdout.write('\n')

            logger.info('Extracting Tiny-Imagenet-200...')
            with ZipFile(project.tiny_imagenet_zip) as zf:
                zf.extractall(project.data)
        
        self.id_dict = {}
        for i, line in enumerate(project.tiny_imagenet_wnids.open('r')):
            self.id_dict[line.replace('\n', '')] = i
        
        logger.info('Loading dataset...')
        print('loading dataset')
        train_data, test_data = [], []
        train_labels, test_labels = [], []
        t = time.time()

        for line in project.tiny_imagenet_val_annotations.open('r'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(imageio.imread(project.tiny_imagenet_val_images / img_name ,pilmode='RGB'))
            test_labels_ = np.array([[0]*200])
            test_labels_[0, self.id_dict[class_id]] = 1
            test_labels += test_labels_.tolist()

        print('finished loading dataset, in {} seconds'.format(time.time() - t))
        train_data, train_labels, test_data, test_labels = np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)

        tensor_x = torch.Tensor(test_data) #torch.cat((torch.Tensor(train_data),torch.Tensor(test_data))) # transform to torch tensor
        tensor_y = torch.Tensor(test_labels) #torch.cat((torch.Tensor(train_labels),torch.Tensor(test_labels)))

        tensor_x = torch.permute(tensor_x, (0, 3, 1, 2))/255
        self._dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your dataset


tiny_imagenet = Datasets_tiny_imagenet()