import torch
import torchvision

from src.utils.pathtools import project
from src.utils.logging import logger

BATCH_SIZE = 1
NAME = 'CIFAR100'


class Datasets():

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

cifar100 = Datasets()
