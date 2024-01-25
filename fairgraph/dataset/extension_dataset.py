import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import os


class SyntheticDataset:
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _open(self):
        with open(self.path, 'rb') as f:
            graph = pickle.load(f)

        return graph


class ExtendDatasetWithIncomeAndEducation(SyntheticDataset):
    def __init__(self, path: str, seed: int = 42):
        self.set_seed(seed)
        self.path = path
        self.graph = self._open()
        self.arr_for_visualization = []

    def _sample_and_add_income(self, attributes: list):
        p_minority_income = [0.9, 0.1]
        p_majority_income = [0.1, 0.9]

        p_current = p_majority_income \
            if attributes['m'] == 1 \
            else p_minority_income

        attributes['income'] = np.random.choice([0, 1], p=p_current)

    def _sample_and_add_education(self, attributes: list):
        p_minority_education = [0.7, 0.3]
        p_majority_education = [0.3, 0.7]

        p_current = p_majority_education \
            if attributes['m'] == 1 \
            else p_minority_education

        attributes['education'] = np.random.choice([0, 1], p=p_current)

    def _add_attributes(self):
        for u in self.graph.nodes:
            attributes = self.graph.nodes[u]
            self._sample_and_add_income(attributes)
            self._sample_and_add_education(attributes)

            self.arr_for_visualization.append(self.graph.nodes[u])

    def _save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.graph, f)

    def process(self):
        self._add_attributes()
        self._save()
        print('Saved!')

    def visualize(self, filename: str = 'graph.png'):
        array = np.array([
            list(d.values()) for d in self.arr_for_visualization
        ])

        df = pd.DataFrame(array, columns=['m', 'income', 'education'])

        counts = df.groupby(['m', 'income', 'education']).size() \
            .reset_index(name='count')

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, group in enumerate(counts['m'].unique()):
            for j, variable in enumerate(['income', 'education']):
                sizes = [
                    counts[counts['m'] == group][
                        counts[variable] == 0
                    ]['count'].sum(),
                    counts[counts['m'] == group][
                        counts[variable] == 1
                    ]['count'].sum()
                ]

                labels = ['0', '1']
                axes[i, j].pie(sizes, labels=labels, autopct='%1.1f%%')
                axes[i, j].set_title(f'm = {group}, {variable}')

        plt.savefig(filename)






# Define a function that takes a file name as an argument
def load_and_print_graph(file_name):
    # Open the file in binary mode and load it as a networkx graph
    with open(file_name, "rb") as f:
        G = pickle.load(f)
    # Print the number of nodes and node features
    print(f"File: {file_name}")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of node features: {len(G.nodes[0])}")
    print("Number of majority nodes: ", len([node for node in G.nodes if G.nodes[node]['m'] == 0]))
    print("Number of minority nodes: ", len([node for node in G.nodes if G.nodes[node]['m'] == 1]))
    print("Number of nodes with high income: ", len([node for node in G.nodes if G.nodes[node]['income'] == 0]))
    # Print the number of edges
    print(f"Number of edges: {G.number_of_edges()}")
    # Define a function that returns the group of a node based on its features
    def get_group(node):
        # For simplicity, assume that the first feature is the group label
        return G.nodes[node]['m']
    # Initialize the counters for inter-group and intra-group edges
    inter_group_edges = 0
    intra_group_edges = 0
    # Loop over all edges and check their groups
    for u, v in G.edges:
        # Get the groups of the end nodes
        u_group = get_group(u)
        v_group = get_group(v)
        # If the groups are different, increment the inter-group counter
        if u_group != v_group:
            inter_group_edges += 1
        # Otherwise, increment the intra-group counter
        else:
            intra_group_edges += 1
    # Print the number of inter-group and intra-group edges
    print(f"Number of inter-group edges: {inter_group_edges}")
    print(f"Number of intra-group edges: {intra_group_edges}")
    print('\n\n\n')

if __name__ == '__main__':
    for file in os.listdir('dataset/artificial/'):
        if file.endswith('.gpickle'):
            load_and_print_graph('dataset/artificial/' + file)

            #dataset = ExtendDatasetWithIncomeAndEducation('dataset/artificial/' + file)
            #dataset.process()

            #print(f'Processing {file}...')

    print('Finished!')
