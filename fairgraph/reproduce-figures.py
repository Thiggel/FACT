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

from dataset import POKEC, NBA, ArtificialSensitiveGraphDataset
from method.Graphair import aug_module


dataset = NBA()

sns.set_style('whitegrid')


def kdeplot(data, label='Original', color='dodgerblue'):
    density = gaussian_kde(data)

    x_values = np.linspace(min(data), max(data), 1000)

    pdf_values = density.evaluate(x_values)

    mode = x_values[np.argmax(pdf_values)]

    plt.plot(x_values, pdf_values, label=label, color=color)

    plt.axvline(mode, color=color, linestyle='--')


data = dataset.node_sensitive_homophily_per_node()
kdeplot(data)

augmentation_module = aug_module(features=dataset.features)

state_dict = torch.load(os.path.join(
    os.getcwd(),
    '../checkpoint',
    'graphair_NBA_alpha20_beta0.9_gamma0.7_lambda1'
))

# TODO:
# 1. make it so that all three datasets are shown side by side
# 2. you only input the HPs, and it'll load all the three files
# 3. make this into a neat class structure with good functions

# Create a new state dictionary
new_state_dict = {}

for key, value in state_dict.items():
    # Check if the key starts with 'aug_model.'
    if key.startswith('aug_model.'):
        # Remove 'aug_model.' from the key
        new_key = key.replace('aug_model.', '', 1)
        new_state_dict[new_key] = value


augmentation_module.load_state_dict(new_state_dict)

adjacency_matrix = scipysp_to_pytorchsp(dataset.adj).to_dense()

dataset.adj = csr_matrix(augmentation_module(adjacency_matrix, dataset.features)[0].detach().numpy())

data = dataset.node_sensitive_homophily_per_node()
kdeplot(data, 'Fair View', 'coral')


plt.legend()
plt.xlabel('Node sensitive homophily')
plt.ylabel('Density')

plt.show()
