import time
import sys
import zipfile

import imageio
import numpy as np
import requests
import torch
import torchvision
from tqdm import tqdm

from src.utils.pathtools import project
from src.utils.logging import logger

TINY_IMAGENET_DOWNLOAD = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
WNIDS_PATH = project.tiny_imagenet / 'wnids.txt'
TRAIN_DIR_PATH = project.tiny_imagenet / 'train'
TRAIN_IMAGE_FORLDER_NAME = 'images'
VAL_DIR_PATH = project.tiny_imagenet / 'val'
VAL_ANNOTATIONS_PATH = VAL_DIR_PATH / 'val_annotations.txt'
VAL_IMAGES_PATH = VAL_DIR_PATH / 'images'
BATCH_SIZE = 32


# Code partly taken from https://github.com/TheAthleticCoder/Tiny-ImageNet-200

class TinyImageNetDataset():

    def __init__(self):
        self._train_dataset: torchvision.datasets.VisionDataset = None
        self._train_loader: torch.utils.data.dataloader.DataLoader = None
        self._train_num_images = 0
        self._val_dataset: torchvision.datasets.VisionDataset = None
        self._val_loader: torch.utils.data.dataloader.DataLoader = None
        self._val_num_images = 0
        self.batch_size = BATCH_SIZE
   
    @property
    def train_dataset(self):
        if self._train_dataset is None:
            self.build_train_dataset()
        return self._train_dataset

    @property
    def train_num_images(self):
        if self._train_num_images == 0:
            self.build_train_dataset()
        return self._train_num_images
        
    @property
    def val_dataset(self):
        if self._val_dataset is None:
            self.build_val_dataset()
        return self._val_dataset

    @property
    def val_num_images(self):
        if self._val_num_images == 0:
            self.build_val_dataset()
        return self._val_num_images
        
    @property
    def train_loader(self):
        """The dataloader returns a tuple (images, labels) where:
        * images is of shape (BATCH_SIZE, 3, 64, 64) -> 64x64 RGB images
        * labels is of shape (BATCH_SIZE, 200) -> one hot encoding of the classes
        """
        if self._train_loader is None:
            logger.info(f'Building dataloader...')
            self._train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return self._train_loader

    @property
    def val_loader(self):
        """The dataloader returns a tuple (images, labels) where:
        * images is of shape (BATCH_SIZE, 3, 64, 64) -> 64x64 RGB images
        * labels is of shape (BATCH_SIZE, 200) -> one hot encoding of the classes
        """
        if self._val_loader is None:
            logger.info(f'Building dataloader...')
            self._val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        return self._val_loader

# -------------------- DOWNLOAD ---------------------

    def check_downloaded(self):
        """Checks that the datasets are correctly downloaded"""
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
                self.check_downloaded()

        else:
            logger.info(f'Tiny-Imagenet-200 found at {project.as_relative(project.tiny_imagenet)}')

# -------------------- BUILDS ---------------------

    def build_train_dataset(self):
        """Builds the train dataset"""
        self.check_downloaded()
    
        self.id_dict = {}
        for i, line in enumerate(WNIDS_PATH.open('r')):
            self.id_dict[line.replace('\n', '')] = i
        
        logger.info('Loading dataset from disk...')
        train_data, train_labels = [], []

        logger.info('Loading train set...')
        for class_path in tqdm(list(TRAIN_DIR_PATH.iterdir())):
            class_id= class_path.name
            for image_path in (class_path / TRAIN_IMAGE_FORLDER_NAME).iterdir():
                train_data.append(imageio.imread(image_path, pilmode='RGB'))
                train_labels_ = np.array([[0]*200])
                train_labels_[0, self.id_dict[class_id]] = 1
                train_labels += train_labels_.tolist()
                self._train_num_images += 1

        logger.info('Converting datasets to torch TensorDataset...')
        train_data, train_labels = np.array(train_data), np.array(train_labels)

        self.tensor_x_train = torch.Tensor(train_data)
        self.tensor_y_train = torch.Tensor(train_labels)

        tensor_x_train = torch.permute(tensor_x_train, (0, 3, 1, 2))/255
        self._train_dataset = torch.utils.data.TensorDataset(self.tensor_x_train,self.tensor_y_train)

        logger.info('Dataset build finished !')
    
    def build_val_dataset(self):
        """Builds the test dataset"""
        self.check_downloaded()
    
        self.id_dict = {}
        for i, line in enumerate(WNIDS_PATH.open('r')):
            self.id_dict[line.replace('\n', '')] = i
        
        logger.info('Loading dataset from disk...')
        val_data, val_labels = [], []

        logger.info('Loading validation set...')
        for line in tqdm(list(VAL_ANNOTATIONS_PATH.open('r'))):
            img_name, class_id = line.split('\t')[:2]
            val_data.append(imageio.imread(VAL_IMAGES_PATH / img_name, pilmode='RGB'))
            val_labels_ = np.array([[0]*200])
            val_labels_[0, self.id_dict[class_id]] = 1
            val_labels += val_labels_.tolist()
            self._val_num_images += 1

        logger.info('Converting datasets to torch TensorDataset...')
        val_data, val_labels = np.array(val_data), np.array(val_labels)

        self.tensor_x_val = torch.Tensor(val_data)
        self.tensor_y_val = torch.Tensor(val_labels)

        tensor_x_val = torch.permute(tensor_x_val, (0, 3, 1, 2))/255
        self._val_dataset = torch.utils.data.TensorDataset(self.tensor_x_val,self.tensor_y_val)

        logger.info('Dataset build finished !')

# -------------------- UTILS ---------------------

    def get_loader_from_tensors(self, img_tensor: torch.Tensor, labels_tensor: torch.TensorType = None) -> torch.utils.data.dataloader.DataLoader:
        """Builds a dataloader from the images tensor and labels tensor
        If no labels are provided, the labels for the validation set are loaded."""
        if labels_tensor is None:
            self.build_val_dataset()
            labels_tensor = self.tensor_y_val
        dataset = torch.utils.data.TensorDataset(img_tensor,labels_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

tiny_imagenet = TinyImageNetDataset()