import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from scipy.sparse import csr_matrix
import os
from scipy.stats import gaussian_kde
import torch
from utils.utils import scipysp_to_pytorchsp

sys.path.insert(0, os.path.abspath('..'))

from dataset import POKEC, NBA, GraphDataset
from method.Graphair import aug_module


class Figures:
    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        lambda_: float,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_

    def kdeplot(subplot, data, label='Original', color='dodgerblue'):
        density = gaussian_kde(data)

        x_values = np.linspace(min(data), max(data), 1000)

        pdf_values = density.evaluate(x_values)

        mode = x_values[np.argmax(pdf_values)]

        subplot.plot(x_values, pdf_values, label=label, color=color)

        subplot.axvline(mode, color=color, linestyle='--')

    def load_augmentation_module(
        self,
        dataset: str = 'NBA'
    ) -> torch.nn.Module:
        augmentation_module = aug_module(features=dataset.features)

        state_dict = torch.load(os.path.join(
            os.getcwd(),
            '../checkpoint',
            f'graphair_{dataset}_{self.alpha}20_' +
            f'{self.beta}0.9_{self.gamma}0.7_{self.lambda_}1'
        ))

        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith('aug_model.'):
                new_key = key.replace('aug_model.', '', 1)
                new_state_dict[new_key] = value

        augmentation_module.load_state_dict(new_state_dict)

        return augmentation_module

    def augment_dataset(
        self,
        augmentation_module: torch.nn.Module,
        dataset: GraphDataset
    ):

        adjacency_matrix = scipysp_to_pytorchsp(dataset.adj).to_dense()

        dataset.adj = csr_matrix(
            augmentation_module(
                adjacency_matrix,
                dataset.features
            )[0].detach().numpy()
        )

    def create_plot(self):
        dataset = NBA()

        sns.set_style('whitegrid')

        ax, _ = plt.subplots(1, 3)

        for index, dataset in enumerate([
            NBA(),
            POKEC(dataset_sample='pokec-z'),
            POKEC(dataset_sample='pokec-n')
        ]):
            data = dataset.node_sensitive_homophily_per_node()
            self.kdeplot(ax[index], data)

            augmentation_module = self.load_augmentation_module()
            self.augment_dataset(augmentation_module, dataset)

            data = dataset.node_sensitive_homophily_per_node()
            self.kdeplot(ax[index], data, 'Fair View', 'coral')

            ax[index].legend()
            ax[index].xlabel('Node sensitive homophily')
            ax[index].ylabel('Density')

        plt.show()
