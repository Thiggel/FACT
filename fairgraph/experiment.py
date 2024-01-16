from .method.Graphair import Graphair, aug_module, GCN, GCN_Body, Classifier
from .utils.constants import Datasets
from .utils.utils import set_device
from .dataset import POKEC, NBA

import time


class Experiment:
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self, dataset_name, device=None, epochs=10_000, test_epochs=1_000,
                 lr=1e-4, weight_decay=1e-5):
        self.device = device if device else set_device()
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset = self.initialize_dataset(dataset_name)

    def initialize_dataset(self, dataset_name):
        if dataset_name == Datasets.NBA:
            return NBA(device=self.device)
        elif dataset_name == Datasets.POKEN:
            return POKEC(device=self.device, dataset_sample='pokec_n')
        elif dataset_name == Datasets.POKEZ:
            return POKEC(device=self.device, dataset_sample='pokec_z')
        else:
            raise Exception(f"Dataset {dataset_name} is not supported")

    def run(self):
        r""" This method runs training and evaluation for a fairgraph model on the given dataset.
        Check :obj:`examples.fairgraph.Graphair.run_graphair_nba.py` for examples on how to run the Graphair model.

        
        :param device: Device for computation.
        :type device: :obj:`torch.device`

        :param model: Defaults to `Graphair`. (Note that at this moment, only `Graphair` is supported)
        :type model: str, optional
        
        :param dataset: The dataset to train on. Should be one of :obj:`dig.fairgraph.dataset.fairgraph_dataset.POKEC` or :obj:`dig.fairgraph.dataset.fairgraph_dataset.NBA`.
        :type dataset: :obj:`object`
        
        :param epochs: Number of epochs to train on. Defaults to 10_000.
        :type epochs: int, optional

        :param test_epochs: Number of epochs to train the classifier while running evaluation. Defaults to 1_000.
        :type test_epochs: int,optional

        :param lr: Learning rate. Defaults to 1e-4.
        :type lr: float,optional

        :param weight_decay: Weight decay factor for regularization. Defaults to 1e-5.
        :type weight_decay: float, optional

        :raise:
            :obj:`Exception` when model is not Graphair. At this moment, only Graphair is supported.
        """

        # Train script
        dataset_name = self.dataset.name

        features = self.dataset.features
        sens = self.dataset.sens
        adj = self.dataset.adj
        idx_sens = self.dataset.idx_sens_train

        # Initialize a model
        self.aug_model = aug_module(features, n_hidden=64, temperature=1, device=self.device).to(self.device)
        self.f_encoder = GCN_Body(in_feats=features.shape[1], n_hidden=64, out_feats=64, dropout=0.1, nlayer=2).to(self.device)
        self.sens_model = GCN(in_feats = features.shape[1], n_hidden=64, out_feats=64, nclass=1).to(self.device)
        self.classifier_model = Classifier(input_dim=64, hidden_dim=64)
        self.model = Graphair(aug_model=self.aug_model, f_encoder=self.f_encoder, sens_model=self.sens_model,
                         classifier_model=self.classifier_model, lr=self.lr, weight_decay=self.weight_decay,
                         dataset=dataset_name).to(self.device)

        # Train the model
        st_time = time.time()
        self.model.fit_whole(epochs=self.epochs, adj=adj, x=features, sens=sens, idx_sens=idx_sens, warmup=0, adv_epoches=1)
        print("Training time: ", time.time() - st_time)

        # Test the model
        self.model.test(adj=adj,
                   features=features,
                   labels=self.dataset.labels,
                   epochs=self.test_epochs,
                   idx_train=self.dataset.idx_train,
                   idx_val=self.dataset.idx_val,
                   idx_test=self.dataset.idx_test,
                   sens=sens)
