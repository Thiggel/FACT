import torch
from torch.utils.data import random_split, Subset
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import random
from graphsaint.minibatch import Minibatch
from torch_geometric.data import download_url
import networkx as nx

from .extension_dataset import SyntheticDataset


class ArtificialSensitiveGraphDataset(SyntheticDataset):
    def __init__(
        self,
        path: str,
        sensitive_attribute: str = 'm',
        target_attribute: str = 'income',
        device: str = 'cpu',
        seed: int = 42
    ) -> None:
        """
        Args:
            path (string): path to the dataset
            sensitive_attribute (string): sensitive attribute
            target_attribute (string): target attribute to be predicted
            device (string): device
            seed (int): seed
        """
        self.name = 'artificial'
        self.path = path
        self.sensitive_attribute = sensitive_attribute
        self.target_attribute = target_attribute
        self.seed = seed
        self.device = device

        self.set_seed(self.seed)

        self.graph = self._open()

        self.splits = self.get_splits()

    @property
    def adj(self):
        return nx.to_scipy_sparse_array(self.graph)

    def _get_unsensitive_features(self, node: dict) -> list:
        return [
            value for key, value in self.graph.nodes[node].items()
            if key != self.sensitive_attribute
            and key != self.target_attribute
        ]

    @property
    def features(self) -> torch.tensor:
        return torch.FloatTensor([
            self._get_unsensitive_features(node)
            for node in self.graph.nodes
        ]).to(self.device)

    @property
    def sens(self) -> torch.tensor:
        return torch.FloatTensor([
            self.graph.nodes[node][self.sensitive_attribute]
            for node in self.graph.nodes
        ]).to(self.device)

    @property
    def labels(self) -> torch.tensor:
        return torch.LongTensor([
            self.graph.nodes[node][self.target_attribute]
            for node in self.graph.nodes
        ]).to(self.device)

    def _subset_to_tensor(self, data: Subset) -> torch.tensor:
        return torch.LongTensor(list(data)).to(self.device)

    def get_splits(
        self,
        train_proportion: float = 0.8,
        val_proportion: float = 0.1,
        test_proportion: float = 0.1,
    ) -> dict:
        indices = list(range(len(self.graph)))

        train_indices, val_indices, test_indices = random_split(
            indices,
            [train_proportion, val_proportion, test_proportion]
        )

        return {
            'train': self._subset_to_tensor(train_indices),
            'val': self._subset_to_tensor(val_indices),
            'test': self._subset_to_tensor(test_indices),
        }

    @property
    def idx_train(self) -> torch.tensor:
        return self.splits['train']

    @property
    def idx_sens_train(self) -> torch.tensor:
        '''
        Those indices within self.idx_train
        that have 1 as the value for the sensitive attribute
        '''
        return self.idx_train[self.sens[self.idx_train] == 1]

    @property
    def idx_val(self) -> torch.tensor:
        return self.splits['val']

    @property
    def idx_test(self) -> torch.tensor:
        return self.splits['test']


class GraphDataset:
    def get_neighbours(self, node):
        return self.adj[node].nonzero()[1]

    def get_neighbours_with_same_sensitive_attribute(self, node, neighbours):
        return neighbours[(self.sens[neighbours] == self.sens[node]).tolist()]

    def get_node_sensitive_homophily(self, node):
        neighbours = self.get_neighbours(node)
        num_neighbours = len(neighbours)
        

        same_sens = self.get_neighbours_with_same_sensitive_attribute(
            node, neighbours
        )
        num_same_sens = len(same_sens)

        return num_same_sens / num_neighbours

    def node_sensitive_homophily_per_node(self):
        return [
            self.get_node_sensitive_homophily(node)
            for node in range(len(self.features))
        ]


class POKEC(GraphDataset):
    r"""Pokec is a social network dataset. Two `different datasets <https://github.com/EnyanDai/FairGNN/tree/main/dataset/pokec>`_ (namely pokec_z and pokec_n) are sampled
        from the original `Pokec dataset <https://snap.stanford.edu/data/soc-pokec.html>`_.

        :param data_path: The url where the dataset is found, defaults to :obj:`https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/pockec/`
        :type data_path: str, optional

        :param root: The path to root directory where the dataset is saved, defaults to :obj:`./dataset/pokec`
        :type root: str, optional

        :param dataset_sample: The sample (should be one of `pokec_z` or `pokec_n`) to be used in choosing the POKEC dataset. Defaults to `pokec_z`
        :type dataset_sample: str, optional
        
        :raises: :obj:`Exception`
            When invalid dataset_sample is provided.
    """
    def __init__(self, 
                data_path='https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/pockec/',
                root='./dataset/pokec',
                dataset_sample='pokec_z',
                device='cpu',
                batch_size=1000):
        self.name = "POKEC_Z"
        self.root = root
        self.dataset_sample = dataset_sample
        if self.dataset_sample=='pokec_z':
            self.dataset = 'region_job'
        elif self.dataset_sample=='pokec_n':
            self.dataset = 'region_job_2'
        else:
            raise Exception('Invalid dataset sample! Should be one of pokec_z or pokec_n')
        self.sens_attr = "region"
        self.predict_attr = "I_am_working_in_field"
        self.label_number = 50000
        self.sens_number = 20000
        self.seed = 20
        self.test_idx=False
        self.data_path = data_path
        self.device=device
        self.batch_size = batch_size
        self.process()
    
    @property
    def raw_paths(self):
        return [f"{self.dataset}.csv",f"{self.dataset}_relationship.txt",f"{self.dataset}.embedding"]
    
    def download(self):
        print('downloading raw files from:', self.data_path)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        for raw_path in self.raw_paths:
            download_url(self.data_path+raw_path,self.root)

    def read_graph(self):
        self.download()
        print(f'Loading {self.dataset} dataset from {os.path.abspath(self.root+"/"+self.raw_paths[0])}')
        # raw_paths[0] will be region_job.csv
        idx_features_labels = pd.read_csv(os.path.abspath(self.root+"/"+self.raw_paths[0]))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(self.sens_attr)
        header.remove(self.predict_attr)


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        # raw_paths[1] will be region_relationship.txt
        edges_unordered = np.genfromtxt(os.path.abspath(self.root+"/"+self.raw_paths[1]), dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        
        random.seed(self.seed)
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.1 * len(label_idx)),self.label_number)]
        idx_val = label_idx[int(0.1 * len(label_idx)):int(0.2 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.2 * len(label_idx)):]

        sens = idx_features_labels[self.sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]

        return 2*(features - min_values).div(max_values-min_values) - 1

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels>1]=1
        sens[sens>0]=1

        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        self.sens = sens.to(self.device)
        self.idx_sens_train = idx_sens_train.long().to(self.device)

        self.adj = adj

        self.create_minibatch()

    def create_minibatch(self):
        ids = np.arange(self.features.shape[0])
        role = {'tr':ids.copy(), 'va': ids.copy(), 'te':ids.copy()}
        train_params = {'sample_coverage': 500}
        train_phase = {'sampler': 'rw', 'num_root': self.batch_size, 'depth': 3, 'end':30}
        self.minibatch = Minibatch(self.adj, self.adj, role, train_params, self.device)
        self.minibatch.set_sampler(train_phase)

class NBA(GraphDataset):
    r'''
        `NBA <https://github.com/EnyanDai/FairGNN/tree/main/dataset/NBA>`_ is an NBA on court performance dataset along salary, social engagement etc.

        :param data_path: The url where the dataset is found, defaults to :obj:`https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/nba/`
        :type data_path: str, optional

        :param root: The path to root directory where the dataset is saved, defaults to :obj:`./dataset/nba`
        :type root: str, optional
    '''
    def __init__(self, 
                data_path='https://github.com/divelab/DIG_storage/raw/main/fairgraph/datasets/nba/',
                root='./dataset/nba',
                device='cpu'):
        self.name = "NBA"
        self.root = root
        self.dataset = 'nba'
        self.sens_attr = "country"
        self.predict_attr = "SALARY"
        self.label_number = 100
        self.sens_number = 500
        self.seed = 20
        self.test_idx=False
        self.data_path = data_path
        self.device=device
        self.process()

    @property
    def raw_paths(self):
        return ["nba.csv","nba_relationship.txt","nba.embedding"]
    
    def download(self):
        print('downloading raw files from:', self.data_path)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        for raw_path in self.raw_paths:
            download_url(self.data_path+raw_path,self.root)

    def read_graph(self):
        self.download()
        print(f'Loading {self.dataset} dataset from {os.path.abspath(self.root+"/"+self.raw_paths[0])}')
        idx_features_labels = pd.read_csv(os.path.abspath(self.root+"/"+self.raw_paths[0]))
        header = list(idx_features_labels.columns)
        header.remove("user_id")

        header.remove(self.sens_attr)
        header.remove(self.predict_attr)


        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[self.predict_attr].values
        

        # build graph
        idx = np.array(idx_features_labels["user_id"], dtype=int)
        idx_map = {j: i for i, j in enumerate(idx)}
        # raw_paths[1] will be nba_relationship.txt
        edges_unordered = np.genfromtxt(os.path.abspath(self.root+"/"+self.raw_paths[1]), dtype=int)

        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # features = normalize(features)
        adj = adj + sp.eye(adj.shape[0])

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(labels)
        # adj = sparse_mx_to_torch_sparse_tensor(adj)

        
        random.seed(self.seed)
        label_idx = np.where(labels>=0)[0]
        random.shuffle(label_idx)
        idx_train = label_idx[:min(int(0.2 * len(label_idx)),self.label_number)]
        idx_val = label_idx[int(0.2 * len(label_idx)):int(0.55 * len(label_idx))]
        if self.test_idx:
            idx_test = label_idx[self.label_number:]
            idx_val = idx_test
        else:
            idx_test = label_idx[int(0.55 * len(label_idx)):]

        sens = idx_features_labels[self.sens_attr].values

        sens_idx = set(np.where(sens >= 0)[0])
        idx_test = np.asarray(list(sens_idx & set(idx_test)))
        sens = torch.FloatTensor(sens)
        idx_sens_train = torch.LongTensor(list(sens_idx))

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train


    def feature_norm(self,features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]

        return 2*(features - min_values).div(max_values-min_values) - 1

    def process(self):
        adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = self.read_graph()
        features = self.feature_norm(features)

        labels[labels>1]=1
        sens[sens>0]=1

        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.idx_train = idx_train.to(self.device)
        self.idx_val = idx_val.to(self.device)
        self.idx_test = idx_test.to(self.device)
        self.sens = sens.to(self.device)
        self.idx_sens_train = idx_sens_train.long().to(self.device)

        self.adj = adj
