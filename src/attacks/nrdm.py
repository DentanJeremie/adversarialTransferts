import typing as t

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.datasets import tiny_imagenet, Datasets_tiny_imagenet

DEFAULT_LAYER = 14
DEFAULT_EPSILON = 256/16
DEFAULT_ATTACK_STEP = 5
DEFAULT_ATTACK_NAME = 'nrdm_vgg_conv33'


class NRDM():

    def __init__(
        self,
        layer = DEFAULT_LAYER,
        epsilon = DEFAULT_EPSILON, 
        nb_attack_step = DEFAULT_ATTACK_STEP, 
        loader: Datasets_tiny_imagenet = tiny_imagenet.loader,
        name: str = DEFAULT_ATTACK_NAME,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.nb_attack_step = nb_attack_step
        self.loader = loader
        self.attack_name = name

        # Model
        logger.info(f'Loading VGG model...')
        self.output_layer = layer
        model_full = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        layers = list(model_full.children())[0][:self.output_layer + 1]
        self.model_conv33 = nn.Sequential(*layers).to(self.device).eval()
        logger.info(f'Successfully loaded VGG on {str(self.device)} and truncated it to layer {self.output_layer}!')

    def attack_step(
            self,
            data: torch.Tensor,
            data_grad: torch.Tensor,
            original_data: torch.Tensor,
        ):
        """Performs one step of the NRDM attack.
        """
        sign_img_grad = data_grad.sign()
        corrupted_data = data + self.epsilon*sign_img_grad

        # Clipping
        corrupted_data = torch.clamp(corrupted_data, original_data - self.epsilon, original_data + self.epsilon)
        corrupted_data = torch.clamp(corrupted_data, 0, 1)

        return corrupted_data

    def run_corruption(self, num_images) -> None:
        """Corrupts `num_images` of the dataset and stores them, both in disk and in self.corrupted_data and self.original_data. 
        """

        self.original_data = None
        self.corrupted_data = None
        self.labels_data = None

        logger.info('Running corruption...')
        for _, (original_data, labels_data) in tqdm(enumerate(self.loader)):
            original_data = t.cast(torch.Tensor, original_data)

            original_data = original_data.to(self.device)
            corrupted_data = torch.clone(original_data)
            original_features = self.model_conv33(original_data)

            for step in range(self.nb_attack_step):
                # First step: we add a random noise to the image
                if step == 0:
                    corrupted_data += self.epsilon/2 * torch.randn_like(original_data)
                    continue

                corrupted_data = corrupted_data.detach() 
                corrupted_data.requires_grad = True

                self.model_conv33.zero_grad()
                corrupted_features = self.model_conv33(corrupted_data)
                loss = F.mse_loss(corrupted_features, original_features)
                loss.backward(retain_graph = True if step < self.nb_attack_step-1 else False)

                data_grad = corrupted_data.grad.data
                corrupted_data = self.attack_step(corrupted_data, data_grad, original_data)

            # Adding to self.original_data and self.corrupted_data
            if self.original_data is None:
                self.original_data = original_data
            else:
                self.original_data = torch.column_stack((self.original_data, original_data))

            if self.corrupted_data is None:
                self.corrupted_data = corrupted_data
            else:
                self.corrupted_data = torch.column_stack((self.corrupted_data, corrupted_data))

            if self.labels_data is None:
                self.labels_data = labels_data
            else:
                self.labels_data = torch.column_stack((self.labels_data, labels_data))

        logger.info('Saving to disk...')
        original_path, corruption_path, labels_path = project.get_new_corruptions_files()
        torch.save(self.original_data, original_path)
        torch.save(self.corrupted_data, corruption_path)
        torch.save(self.labels_data, labels_path)
        
nrdm14 = NRDM(14)

def show(imgs):
    """Shows a torch tensor as image or a batch of image or a list of image"""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(nrows = len(imgs), ncols=2, squeeze=False)
    for i, img in enumerate(imgs):
        img1, img2 = img
        img1 = img1.detach()
        img2 = img2.detach()
        img1 = torchvision.transforms.functional.to_pil_image(img1)
        img2 = torchvision.transforms.functional.to_pil_image(img2)
        axs[i, 0].imshow(np.asarray(img1))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 1].imshow(np.asarray(img2))
        axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig("output/nrdm14.png")

# show(nrdm14.run_corruption(2))
        