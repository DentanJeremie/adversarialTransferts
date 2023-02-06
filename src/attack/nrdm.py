"""
This script performs the regular NRDM attack on images.
"""

import pathlib
import typing as t

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.datasets import tiny_imagenet, TinyImageNetDataset

DEFAULT_LAYER = 14
DEFAULT_EPSILON = 16/256
DEFAULT_ATTACK_STEP = 5
DEFAUTL_NB_PLOT = 5
DEFAULT_SAVE_ORIGINALS = False
DEFAULT_SAVE_LABELS = False
VERBOSE = 10
DEFAULT_ATTACK_NAME = 'nrdm_vgg_conv33'
DEFAULT_FULL_MODEL = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)

LAYERS = [7, 14, 21]
NB_ATTACK_STEPS = [3, 5, 7, 10]
ATTACK_NAME = ['vgg_conv22_{nb}steps', 'vgg_conv33_{nb}steps', 'vgg_conv43_{nb}steps']
ATTACK_FINAL_NAMES = [
    'vgg_conv22_3steps',
    'vgg_conv22_5steps',
    'vgg_conv22_7steps',
    'vgg_conv22_10steps',
    'vgg_conv33_3steps',
    'vgg_conv33_5steps',
    'vgg_conv33_7steps',
    'vgg_conv33_10steps',
    'vgg_conv43_3steps',
    'vgg_conv43_5steps',
    'vgg_conv43_7steps',
    'vgg_conv43_10steps',
]


class NRDM():

    def __init__(
        self,
        layer = DEFAULT_LAYER,
        epsilon = DEFAULT_EPSILON, 
        nb_attack_step = DEFAULT_ATTACK_STEP, 
        attack_name: str = DEFAULT_ATTACK_NAME,
        model_full: torch.nn.Module = DEFAULT_FULL_MODEL,
        save_originals: bool = DEFAULT_SAVE_ORIGINALS,
        save_labels: bool = DEFAULT_SAVE_LABELS,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.nb_attack_step = nb_attack_step
        self.attack_name = attack_name
        self.save_originals = save_originals
        self.save_labels = save_labels

        # Dataset
        self.dataset: TinyImageNetDataset = tiny_imagenet

        # Model
        logger.info(f'Loading the full model...')
        self.output_layer = layer
        model_full = model_full
        layers = list(model_full.children())[0][:self.output_layer + 1]
        self.model_truncated = nn.Sequential(*layers).to(self.device).eval()
        logger.info(f'Successfully loaded the full model on {str(self.device)} and truncated it to layer {self.output_layer}!')


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

    def run_corruption(self, num_images = -1) -> None:
        """Corrupts `num_images` of the dataset and stores them, both in disk and in self.corrupted_data and self.original_data. 

        if `num_images`== -1, the corruption is run on the whole dataset.
        """
        if num_images == -1:
            num_images = self.dataset.val_num_images
        else:
            num_images = min(num_images, self.dataset.val_num_images)

        self.original_data = None
        self.corrupted_data = None
        self.labels_data = None

        logger.info(f'Running corruption of {num_images} images')
        for batch_number, (original_data, labels_data) in enumerate(self.dataset.val_loader):
            if batch_number % VERBOSE == 0:
                logger.debug(f'Starting batch {batch_number+1}/{int(np.ceil(num_images/self.dataset.batch_size))}')
            original_data = t.cast(torch.Tensor, original_data)
            labels_data = t.cast(torch.Tensor, labels_data)

            original_data = original_data.to(self.device)
            corrupted_data = torch.clone(original_data)
            original_features = self.model_truncated(original_data)

            for step in range(self.nb_attack_step):
                # First step: we add a random noise to the image
                if step == 0:
                    corrupted_data += self.epsilon/2 * torch.randn_like(original_data)
                    continue

                corrupted_data = corrupted_data.detach() 
                corrupted_data.requires_grad = True

                self.model_truncated.zero_grad()
                corrupted_features = self.model_truncated(corrupted_data)
                loss = F.mse_loss(corrupted_features, original_features)
                loss.backward(retain_graph = True if step < self.nb_attack_step-1 else False)

                data_grad = corrupted_data.grad.data
                corrupted_data = self.attack_step(corrupted_data, data_grad, original_data)

            # Adding to self.original_data and self.corrupted_data
            if self.original_data is None:
                self.original_data = original_data.detach().cpu()
            else:
                self.original_data = torch.cat((self.original_data, original_data.detach().cpu()), dim=0)

            if self.corrupted_data is None:
                self.corrupted_data = corrupted_data.detach().cpu()
            else:
                self.corrupted_data = torch.cat((self.corrupted_data, corrupted_data.detach().cpu()), dim=0)

            if self.labels_data is None:
                self.labels_data = labels_data.detach().cpu()
            else:
                self.labels_data = torch.cat((self.labels_data, labels_data.detach().cpu()), dim=0)

            # Checking the number of images
            if num_images>= 0 and self.original_data.size(0) >= num_images:
                logger.info(f'Enough images corrupted for what was asked ({num_images})!')
                self.original_data = self.original_data[:num_images]
                self.corrupted_data = self.corrupted_data[:num_images]
                self.labels_data = self.labels_data[:num_images]
                break

        logger.info('Corruption finished!')

        logger.info('Saving to disk...')
        original_path, corruption_path, labels_path, plot_path = project.get_new_corruptions_files(self.attack_name)
        torch.save(self.corrupted_data, corruption_path)
        if self.save_originals:
            torch.save(self.original_data, original_path)
        if self.save_labels:
            torch.save(self.labels_data, labels_path)
        self.plot_corruptions(DEFAUTL_NB_PLOT, plot_path)
        logger.info(f'Saved to {project.as_relative(corruption_path)}')

    def plot_corruptions(self, num_images: int, save_path: pathlib.Path):
        """Plots a torch tensor as image or a batch of image or a list of image"""
        logger.info(f'Plotting {num_images} example of corruptions...')
        images = [
            (self.original_data[index], self.corrupted_data[index])
            for index in range(min(num_images, self.original_data.size(0)))
        ]
        fig, axs = plt.subplots(nrows = len(images), ncols=2, squeeze=False)
        for i, img in enumerate(images):
            img1, img2 = img
            img1 = img1.detach()
            img2 = img2.detach()
            img1 = torchvision.transforms.functional.to_pil_image(img1)
            img2 = torchvision.transforms.functional.to_pil_image(img2)
            axs[i, 0].imshow(np.asarray(img1))
            axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[i, 1].imshow(np.asarray(img2))
            axs[i, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(save_path)

    def clear_gpu(self):
        """Clears the GPU memory, and brings back everything to CPU"""
        logger.info('Clearing GPU memory and bringing everything back to CPU')
        self.model_truncated.cpu()
        self.original_data.cpu()
        self.corrupted_data.cpu()
        self.labels_data.cpu()
        torch.cuda.empty_cache()


def main():
    for layer, name in zip(LAYERS, ATTACK_NAME):
        for nb_step in NB_ATTACK_STEPS:
            name_eddited = name.format(nb=nb_step)
            logger.info(f'Run on layer {layer} with {nb_step} steps -> {name_eddited}')
            attacker = NRDM(
                layer=layer,
                nb_attack_step=nb_step,
                attack_name=name_eddited
            )
            attacker.run_corruption(-1)

            del attacker
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
        