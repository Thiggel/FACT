from .method.Graphair import Graphair, aug_module, GCN, GCN_Body, Classifier
from .utils.constants import Datasets
from .utils.utils import set_device
from .dataset import POKEC, NBA

import time

# TODO: go through all the models and replace hardcoded hyperparemters with arguments, then add to hyperparams file

class Experiment:
    r"""
    This class instantiates Graphair model and implements method to train and evaluate.
    """

    def __init__(self, dataset_name, device=None, epochs=10_000, test_epochs=1_000,
                 lr=1e-4, weight_decay=1e-5, g_temperature=1.0, g_hidden=64,
                 f_hidden=64, f_layers=2, f_dropout=0.1, f_output_features=64,
                 k_hidden=64, k_output_features=64, c_hidden=64, c_input=64,
                 warmup=0):
        self.device = device if device else set_device()
        self.epochs = epochs
        self.test_epochs = test_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dataset = self.initialize_dataset(dataset_name)

        # Augmentation model g hyperparameters
        self.g_temperature = g_temperature
        self.g_hidden = g_hidden

        # Encoder model f hyperparameters
        self.f_hidden = f_hidden
        self.f_dropout = f_dropout
        self.f_layers = f_layers
        self.f_output_features = f_output_features

        # Adversary model k hyperparameters
        self.k_hidden = k_hidden
        self.k_output_features = k_output_features

        # Classifier model hyperparameters
        self.c_hidden = c_hidden
        self.c_input = c_input

        # Graphair model hyperparameters
        self.warmup = warmup

    def initialize_dataset(self, dataset_name):
        if dataset_name == Datasets.NBA:
            return NBA(device=self.device)
        elif dataset_name == Datasets.POKEC_N:
            return POKEC(device=self.device, dataset_sample='pokec_n')
        elif dataset_name == Datasets.POKEC_Z:
            return POKEC(device=self.device, dataset_sample='pokec_z')
        else:
            raise Exception(f"Dataset {dataset_name} is not supported. Available datasets are: {[Datasets.POKEC_Z, Datasets.POKEC_N, Datasets.NBA]}")

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

        # Initialize augmentation model g
        self.aug_model = aug_module(
            features=features,
            n_hidden=self.g_hidden,
            temperature=self.g_temperature,
            device=self.device
            ).to(self.device)

        # Initialize encoder model f
        self.f_encoder = GCN_Body(
            in_feats=features.shape[1],
            n_hidden=self.f_hidden,
            out_feats=self.f_output_features,
            dropout=self.f_dropout,
            nlayer=self.f_layers
            ).to(self.device)

        # Initialize adversary model k
        self.sens_model = GCN(
            in_feats=features.shape[1],
            n_hidden=self.k_hidden,
            out_feats=self.k_output_features,
            nclass=1
            ).to(self.device)

        # Initialize classifier for testing
        self.classifier_model = Classifier(input_dim=self.c_input, hidden_dim=self.c_hidden)

        # Initialize the Graphair model
        self.model = Graphair(
            aug_model=self.aug_model,
            f_encoder=self.f_encoder,
            sens_model=self.sens_model,
            classifier_model=self.classifier_model,
            lr=self.lr, # TODO: add a separate lr for classifier model
            weight_decay=self.weight_decay, # TODO: add a separate weight_decay for classifier model
            dataset=dataset_name
            ).to(self.device)

        # Train the model
        st_time = time.time()
        self.model.fit_whole(
            epochs=self.epochs,
            adj=adj,
            x=features,
            sens=sens,
            idx_sens=idx_sens,
            warmup=self.warmup,
            adv_epoches=1) # TODO: figure out what adv_epochs is
        print("Training time: ", time.time() - st_time)

        # Test the model
        self.model.test(
            adj=adj,
            features=features,
            labels=self.dataset.labels,
            epochs=self.test_epochs,
            idx_train=self.dataset.idx_train,
            idx_val=self.dataset.idx_val,
            idx_test=self.dataset.idx_test,
            sens=sens)
