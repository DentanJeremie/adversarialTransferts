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
from src.utils.datasets import cifar100, tiny_imagenet

DEFAULT_LAYER = 14
DEFAULT_EPSILON = 256/16
DEFAULT_ATTACK_STEP = 5
DEFAULT_DATA = cifar100

class NRDM():

    def __init__(self, layer = DEFAULT_LAYER, epsilon = DEFAULT_EPSILON, nb_attack_step = DEFAULT_ATTACK_STEP, dataset = DEFAULT_DATA):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.nb_attack_step = nb_attack_step

        # Model
        logger.info(f'Loading VGG model...')
        self.output_layer = layer
        model_full = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.DEFAULT)
        layers = list(model_full.children())[0][:self.output_layer + 1]
        self.model_conv33 = nn.Sequential(*layers).to(self.device).eval()
        logger.info(f'Successfully loaded VGG on {str(self.device)} and truncated it to layer {self.output_layer}!')

        # Dataset
        self.loader = dataset.loader

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

    def run_corruption(self, num_images = 1) -> t.List[t.Tuple[torch.Tensor, torch.Tensor]]:
        """Returns a list of tuple (original_image, corrupted_image) after performing the attack.
        """
        result = list()

        for index, (data, _) in enumerate(self.loader):
            data = t.cast(torch.Tensor, data)

            data = data.to(self.device)
            corrupted_data = torch.clone(data)
            original_features = self.model_conv33(data)

            for step in range(self.nb_attack_step):
                # First step: we add a random noise to the image
                if step == 0:
                    corrupted_data += self.epsilon/2 * torch.randn_like(data)
                    continue

                corrupted_data = corrupted_data.detach() 
                corrupted_data.requires_grad = True

                self.model_conv33.zero_grad()
                corrupted_features = self.model_conv33(corrupted_data)
                loss = F.mse_loss(corrupted_features, original_features)
                loss.backward(retain_graph = True if step < self.nb_attack_step-1 else False)

                data_grad = corrupted_data.grad.data
                corrupted_data = self.attack_step(corrupted_data, data_grad, data)

        
            for index in range(data.size(0)):
                result.append(
                    (data[index,:,:,:], corrupted_data[index,:,:,:])
                )

                if len(result) == num_images:
                    return result

        return result

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

show(nrdm14.run_corruption(2))

        