from .method.Graphair import Graphair, aug_module, GCN, GCN_Body, Classifier
from .utils.constants import Datasets
from .utils.utils import set_device
from .dataset import POKEC, NBA

import time

# TODO: go through all the models and replace hardcoded hyperparemters with arguments, then add to hyperparams file

class Experiment:
    """
    Creates an experiment with the specified hyperparameters. Instantiates
    Graphair model and implements method to train and evaluate.
    """

    def __init__(self, dataset_name, device=None, epochs=10_000, test_epochs=1_000,
                 lr=1e-4, weight_decay=1e-5, g_temperature=1.0, g_hidden=64,
                 f_hidden=64, f_layers=2, f_dropout=0.1, f_output_features=64,
                 k_hidden=64, k_output_features=64, c_hidden=64, c_input=64,
                 warmup=0):
        """
        Initializes an Experiment class instance.

        Args:
            dataset_name (str): the name of the dataset to use
            device (str): the device to use. Default: None, which
                selects the best available device
            epochs (int): number of training epochs #TODO: update this
            test_epochs (int): number of testing epochs #TODO: update this
            lr (float): learning rate for ... #TODO: update this
            weight_decay (float): weight decay factor for ... #TODO: update this
            g_temperature (float): temperature of augmentation model g
            ... #TODO: finish docstring

        """
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
        """ Runs training and evaluation for a fairgraph model on the given dataset. """

        # Initialize augmentation model g
        self.aug_model = aug_module(
            features=self.dataset.features,
            n_hidden=self.g_hidden,
            temperature=self.g_temperature,
            device=self.device
            ).to(self.device)

        # Initialize encoder model f
        self.f_encoder = GCN_Body(
            in_feats=self.dataset.features.shape[1],
            n_hidden=self.f_hidden,
            out_feats=self.f_output_features,
            dropout=self.f_dropout,
            nlayer=self.f_layers
            ).to(self.device)

        # Initialize adversary model k
        self.sens_model = GCN(
            in_feats=self.dataset.features.shape[1],
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
            dataset=self.dataset.name
            ).to(self.device)

        # Train the model
        st_time = time.time()
        self.model.fit_whole(
            epochs=self.epochs,
            adj=self.dataset.adj,
            x=self.dataset.features,
            sens=self.dataset.sens,
            idx_sens=self.dataset.idx_sens_train,
            warmup=self.warmup,
            adv_epoches=1) # TODO: figure out what adv_epochs is
        print("Training time: ", time.time() - st_time)

        # Test the model
        self.model.test(
            adj=self.dataset.adj,
            features=self.dataset.features,
            labels=self.dataset.labels,
            epochs=self.test_epochs,
            idx_train=self.dataset.idx_train,
            idx_val=self.dataset.idx_val,
            idx_test=self.dataset.idx_test,
            sens=self.dataset.sens)
