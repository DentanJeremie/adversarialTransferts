import torch
from PIL import Image
import numpy as np

import pathlib
import typing as t

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from lucent.optvis import render, param, transform, objectives
from lucent.modelzoo import inceptionv1, vgg16
from lucent.misc.io import show
from lucent.optvis.objectives import wrap_objective

from src.utils.pathtools import project
from src.utils.logging import logger
from src.utils.datasets import tiny_imagenet, TinyImageNetDataset


DEFAULT_LAYERS = ['features_30']

DEFAULT_EPSILON = 16/256
DEFAULT_ATTACK_STEP = 5
DEFAUTL_NB_PLOT = 5
VERBOSE = 10
DEFAULT_ATTACK_NAME = 'Decorrelate_FFT'
DEFAULT_MODEL = vgg16(pretrained=True)

LAYERS = [['features_30']]
NB_ATTACK_STEPS = [2, 3, 5, 7, 10]
ATTACK_NAME = ['Decorrelate_FFT_{nb}steps']
ATTACK_FINAL_NAMES = [
    'Decorrelate_FFT_2steps',
    'Decorrelate_FFT_3steps',
    'Decorrelate_FFT_5steps',
    'Decorrelate_FFT_7steps',
    'Decorrelate_FFT_10steps',
]


class Decorrelate_FFT_attack():

    def __init__(
        self,
        layers = DEFAULT_LAYERS,
        epsilon = DEFAULT_EPSILON, 
        nb_attack_step = DEFAULT_ATTACK_STEP, 
        attack_name: str = DEFAULT_ATTACK_NAME,
        model: torch.nn.Module = DEFAULT_MODEL,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.nb_attack_step = nb_attack_step
        self.attack_name = attack_name

        # Dataset
        self.dataset: TinyImageNetDataset = tiny_imagenet

        # Model
        logger.info(f'Loading the full model...')
        self.output_layers = layers
        model = model
        logger.info(f'Successfully loaded the model on {str(self.device)}!')

    def attack_param(self, original_data, decorrelate=True, fft=True):
        params, image = param.image(*original_data.shape[:2], decorrelate=decorrelate, fft=fft)
        def inner():
            attack_input = image()[0]
            original_data = torch.permute(original_data, [2, 1, 0])
            return torch.stack([attack_input + original_data + self.epsilon/2 * torch.randn_like(original_data), original_data])
        return params, inner

    def MSE(a, b):
        return F.mse_loss(a, b)

    @wrap_objective()
    def features_difference(self, activation_loss_f=MSE):
        def inner(T):
            original_features = [T(layer)[1] for layer in self.output_layers]
            corrupted_features = [T(layer)[0] for layer in self.output_layers]
            losses = [activation_loss_f(a, b) for a, b in zip(original_features, corrupted_features)]
            return sum(losses)

        return inner

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
            
            param_f = lambda: self.attack_param(original_data, decorrelate=True, fft=True)

            objective = self.features_difference(self.output_layers, difference_to=1)
            objective.description = "Attack Loss"

            corrupted_data = render.render_vis(self.model, -objective, param_f, show_inline=True, thresholds = [100])[0][0]

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
        torch.save(self.original_data, original_path)
        torch.save(self.corrupted_data, corruption_path)
        torch.save(self.labels_data, labels_path)
        self.plot_corruptions(DEFAUTL_NB_PLOT, plot_path)
        logger.info(f'Saved to {project.as_relative(plot_path)}')

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
            attacker = Decorrelate_FFT_attack(
                layers=[layer],
                nb_attack_step=nb_step,
                attack_name=name_eddited
            )
            attacker.run_corruption(-1)

            del attacker
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
        