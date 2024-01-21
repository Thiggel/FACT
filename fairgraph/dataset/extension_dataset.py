import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch


class SyntheticDataset:
    def set_seed(seed):
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






if __name__ == '__main__':
    dataset = ExtendDatasetWithIncomeAndEducation('dataset/artificial/DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-hMM0.2-hmm0.2-ID0.gpickle')
    dataset.process()
    dataset.visualize('pie-chart.png')
