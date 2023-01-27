import typing as t 

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pathtools import project, CORRUPTED_FILES_SUFFIX
from src.utils.logging import logger
from src.utils.datasets import tiny_imagenet, TinyImageNetDataset
from src.classifiers_evaluation.models import DenseNet, ResNet, VGG
from src.attack.nrdm import ATTACK_FINAL_NAMES

"""
    We use models from models.py with frozen backbones that we fine tune on the tiny imagenet dataset.
    Normally we should train on tiny imagenet training set and evaluate on validation set.
    We evaluate the accuracy of the fine tuned models on the corrupted datasets.
    We then look at how corruptions coming from different models affect this accuracy.
"""

DEFAULT_OPTIMIZER_LR = 0.001
DEFAULT_LOSS = nn.CrossEntropyLoss()
DEFAULT_EPOCHS = 1
MODELS = [DenseNet(), ResNet(), VGG()]
MODELS_NAMES = ['DenseNet', 'ResNet', 'VGG']
RESULTS_HEADER = ['Dataset', 'Accuracy']


class AdverseEvaluator():

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        *,
        dataset: TinyImageNetDataset = tiny_imagenet,
        loss: nn.modules.loss = DEFAULT_LOSS,
        epochs: int = DEFAULT_EPOCHS,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.dataset = dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=DEFAULT_OPTIMIZER_LR)
        self.loss = loss
        self.epochs = epochs

        # Checking model on disk
        self.model_trained = False
        possible_existing_file = project.get_lastest_trained_classifiers_file(self.model_name)
        if possible_existing_file is not None:
            logger.info(f'Found a trained version at {project.as_relative(possible_existing_file)}')
            try:
                self.model.load_state_dict(torch.load(possible_existing_file))
                logger.info('Successfully loaded the trained model from disk!')
                self.model_trained = True
            except RuntimeError:
                logger.info('Unable to load the model from the disk, preparing for training')

    def train(self):
        """
        Trains the model of `epochs` on the train set of the datasets.
        """
        logger.info(f'Starting training on {self.epochs} epochs')
        for epoch in range(1, self.epochs + 1):
            for batch_idx, (data, target) in enumerate(self.dataset.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target.argmax(dim=1))
                loss.backward()
                self.optimizer.step()
                if batch_idx % 20 == 0:
                    logger.info(
                        f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.dataset.train_dataset)} '
                        f'({100. * batch_idx / len(self.dataset.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        logger.info('Training: done!')
        logger.info('Saving model to disk...') 
        torch.save(self.model.state_dict(), project.get_new_trained_classifier_file(self.model_name))
        self.model_trained = True

    def evaluate(self) -> t.List[t.Tuple[str, float]]:
        """Evaluates the model on the corrupted images. 
        Returns a list of tuples (dataset_name, accuracy_on_this_dataset)."""
        if not self.model_trained:
            self.train()

        results = list()

        logger.info('Evaluating on original images...')
        original_loader = self.dataset.val_loader
        prop_correct = self.evaluate_one_loader(original_loader)
        results.append(['original', prop_correct])
        logger.info(f'Accuracy on original images: {prop_correct}')

        for attack_name in ATTACK_FINAL_NAMES:
            logger.info(f'Reading corrupted data for attack {attack_name}...')
            if project.get_lastest_corruptions_file(attack_name, CORRUPTED_FILES_SUFFIX) is None:
                logger.info('No data found, skipping')
                continue

            loader = tiny_imagenet.get_loader_from_tensors(
                torch.load(project.get_lastest_corruptions_file(attack_name, CORRUPTED_FILES_SUFFIX))
            )
            prop_correct = self.evaluate_one_loader(loader)
            results.append([attack_name, prop_correct])
            logger.info(f'Accuracy on {attack_name}: {prop_correct}')

        result_path = project.get_new_classification_result_path(self.model_name, self.epochs)
        logger.info(f'Saving results on {project.as_relative(result_path)}')
        pd.DataFrame(results, columns=RESULTS_HEADER).to_csv(result_path)

        return results

    def evaluate_one_loader(self, loader) -> float:
        """Returns the proportion of correctly classified images for one given loader.
        """
        correct = 0
        length = 0
        self.model.eval()
        for data, target in loader:
            length += data.size(0)
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.argmax(dim=1, keepdim=True).view_as(pred)).sum().item()
        return correct / length

def main():
    for model, model_name in zip(MODELS, MODELS_NAMES):
        evaluator = AdverseEvaluator(model, model_name)
        evaluator.evaluate()

if __name__ == '__main__':  
    main()
    