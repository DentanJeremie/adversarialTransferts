import time
import sys
import zipfile

import imageio
import numpy as np
import requests
import torch
import torchvision

from src.utils.pathtools import project
from src.utils.logging import logger

TINY_IMAGENET_DOWNLOAD = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
WNIDS_PATH = project.tiny_imagenet / 'wnids.txt'
VAL_DIR_PATH = project.tiny_imagenet / 'val'
VAL_ANNOTATIONS_PATH = VAL_DIR_PATH / 'val_annotations.txt'
VAL_IMAGES_PATH = VAL_DIR_PATH / 'images'
BATCH_SIZE = 32


# Code partly taken from https://github.com/TheAthleticCoder/Tiny-ImageNet-200

class Datasets_tiny_imagenet():

    def __init__(self):
        self._dataset: torchvision.datasets.VisionDataset = None
        self._loader: torch.utils.data.dataloader.DataLoader = None
        self._num_images = 0
        self.batch_size = BATCH_SIZE
   
    @property
    def dataset(self):
        if self._dataset is None:
            self.build_dataset()
        return self._dataset

    @property
    def num_images(self):
        if self._num_images == 0:
            self.build_dataset()
        return self._num_images
        
    @property
    def loader(self):
        """The dataloader returns a tuple (images, labels) where:
        * images is of shape (BATCH_SIZE, 3, 64, 64) -> 64x64 RGB images
        * labels is of shape (BATCH_SIZE, 200) -> one hot encoding of the classes
        """
        if self._loader is None:
            logger.info(f'Building dataloader...')
            self._loader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=False)
        return self._loader

# -------------------- DOWNLOAD ---------------------

    def build_dataset(self):
        """Builds the datasets"""
        
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
                        sys.stdout.write("\rProgression: [%s%s]" % ('=' * done, ' ' * (50-done)) )    
                        sys.stdout.flush()

                sys.stdout.write('\n')

            logger.info('Extracting Tiny-Imagenet-200... (this can take several minutes)')
            try:
                with zipfile.ZipFile(project.tiny_imagenet_zip) as zf:
                    zf.extractall(project.data)
            except zipfile.BadZipFile:
                logger.info(f'Found corrupted .zip file, deleting it an trying again...')
                project.tiny_imagenet_zip.unlink()
                self.build_dataset()
        
        self.id_dict = {}
        for i, line in enumerate(WNIDS_PATH.open('r')):
            self.id_dict[line.replace('\n', '')] = i
        
        logger.info('Loading dataset from disk...')
        val_data, val_labels = [], []

        for line in VAL_ANNOTATIONS_PATH.open('r'):
            img_name, class_id = line.split('\t')[:2]
            val_data.append(imageio.imread(VAL_IMAGES_PATH / img_name, pilmode='RGB'))
            val_labels_ = np.array([[0]*200])
            val_labels_[0, self.id_dict[class_id]] = 1
            val_labels += val_labels_.tolist()
            self._num_images += 1

        logger.info('Converting datasets to torch TensorDataset...')
        val_data, val_labels = np.array(val_data), np.array(val_labels)

        tensor_x = torch.Tensor(val_data)
        tensor_y = torch.Tensor(val_labels)

        tensor_x = torch.permute(tensor_x, (0, 3, 1, 2))/255
        self._dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)

        logger.info('Dataset build finished !')

tiny_imagenet = Datasets_tiny_imagenet()