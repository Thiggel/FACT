import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from scipy.sparse import csr_matrix
import os
from scipy.stats import gaussian_kde
import torch
from utils.utils import scipysp_to_pytorchsp
import argparse

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
        dataset: str
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.dataset = dataset

    def kdeplot(self, data, label='Original', color='royalblue'):
        density = gaussian_kde(data)

        x_values = np.linspace(min(data), max(data), 1000)

        pdf_values = density.evaluate(x_values)

        mode = x_values[np.argmax(pdf_values)]

        plt.plot(x_values, pdf_values, label=label, color=color)

        plt.axvline(mode, color=color, linestyle='--')

    def get_filename(self) -> str:
        return (
            f'graphair_{self.dataset}_alpha{self.alpha}_' +
            f'beta{self.beta}_gamma{self.gamma}_lambda{self.lambda_}'
        )

    def load_augmentation_module(
        self,
        dataset: GraphDataset
    ) -> torch.nn.Module:
        augmentation_module = aug_module(
            features=dataset.features,
            normalize=True
        )

        try:
            state_dict = torch.load(os.path.join(
                os.getcwd(),
                '../checkpoint',
                self.get_filename()
            ))
        except FileNotFoundError:
            print('Checkpoint not found')
            exit()

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

        new_adjacency_matrix, new_features, _ = augmentation_module(
            adjacency_matrix,
            dataset.features
        )

        dataset.adj = csr_matrix(new_adjacency_matrix.detach().numpy())
        dataset.features = new_features.detach().numpy()

    def init_dataset(self) -> GraphDataset:
        try:
            return {
                'NBA': lambda: NBA(),
                'POKEC-Z': lambda: POKEC(dataset_sample='pokec_z'),
                'POKEC-N': lambda: POKEC(dataset_sample='pokec_n')
            }[self.dataset]()
        except:
            print('Invalid dataset')
            exit()

    def nsh_plot(self):
        sns.set_style('whitegrid')

        dataset = self.init_dataset()

        data = dataset.node_sensitive_homophily_per_node()

        self.kdeplot(data)

        augmentation_module = self.load_augmentation_module(dataset)
        self.augment_dataset(augmentation_module, dataset)

        data = dataset.node_sensitive_homophily_per_node()
        self.kdeplot(data, 'Fair View', 'coral')

        plt.legend()
        plt.xlabel('Node sensitive homophily')
        plt.ylabel('Density')

        plt.savefig(
            os.path.join(
                os.getcwd(),
                '../experiments/plots',
                self.get_filename() + '_nsh.svg'
            )
        )
        plt.close()

    def correlation_plot(self):
        sns.set_style('whitegrid')
        dataset = self.init_dataset()

        correlation = dataset.get_correlation_sens()
        sort_indices = np.argsort(correlation)[::-1]
        correlation = correlation[sort_indices]

        augmentation_module = self.load_augmentation_module(dataset)
        self.augment_dataset(augmentation_module, dataset)

        correlation_aug = dataset.get_correlation_sens()
        correlation_aug = correlation_aug[sort_indices]

        x = np.arange(len(correlation[0:10]))

        plt.bar(x - 0.2, correlation[:10], 0.4, label='Original', color='royalblue')

        plt.bar(x + 0.2, correlation_aug[:10], 0.4, label='Fair view', color='coral')

        plt.xlabel('Feature index')
        plt.ylabel('Spearman correlation')
        plt.legend()

        plt.savefig(
            os.path.join(
                os.getcwd(),
                '../experiments/plots',
                self.get_filename() + '_correlation.svg'
            )
        )
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--lambda_', type=float)
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    figures = Figures(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_=args.lambda_,
        dataset=args.dataset
    )

    figures.correlation_plot()
    figures.nsh_plot()
