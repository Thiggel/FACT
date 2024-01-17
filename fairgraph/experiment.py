from .method.Graphair import Graphair, aug_module, GCN, GCN_Body, Classifier
from .utils.constants import Datasets
from .utils.utils import set_device, set_seed
from .dataset import POKEC, NBA

import time

# TODO: go through all the models and replace hardcoded hyperparemters with arguments, then add to hyperparams file


class Experiment:
    """
    Creates an experiment with the specified hyperparameters. Instantiates
    Graphair model and implements method to train and evaluate.
    """

    def __init__(
        self,
        dataset_name,
        device=None,
        verbose=False,
        epochs=10_000,
        test_epochs=1_000,
        seed=42,
        weight_decay=1e-5,
        g_temperature=1.0,
        g_hidden=64,
        g_dropout=0.1,
        g_nlayer=1,
        mlpx_dropout=0.1,
        f_hidden=64,
        f_layers=2,
        f_dropout=0.1,
        f_output_features=64,
        k_hidden=64,
        k_output_features=64,
        k_dropout=0.1,
        k_nlayer=2,
        c_hidden=64,
        c_input=64,
        warmup=0,
        alpha=20,
        beta=0.9,
        gamma=0.7,
        lam=1,
        k_lr=1e-4,
        c_lr=1e-3,
        g_lr=1e-4,
        g_warmup_lr=1e-3,
        f_lr=1e-4,
        graphair_temperature=0.07
    ):
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
        self.dataset = self.initialize_dataset(dataset_name)
        self.verbose = verbose

        # Set a seed for reproducibility
        set_seed(seed)

        # Trainig hyperparameters
        self.warmup = warmup
        self.epochs = epochs
        self.test_epochs = test_epochs

        # Augmentation model g hyperparameters
        self.g_hyperparams = {
            "n_hidden": g_hidden,
            "temperature": g_temperature,
            "nlayer": g_nlayer,
            "dropout": g_dropout,
            "mlpx_dropout": mlpx_dropout,
        }

        # Encoder model f hyperparameters
        self.f_hyperparams = {
            "n_hidden": f_hidden,
            "out_feats": f_output_features,
            "dropout": f_dropout,
            "nlayer": f_layers
        }

        # Adversary model k hyperparameters
        self.k_hyperparams = {
            "n_hidden": k_hidden,
            "out_feats": k_output_features,
            "dropout": k_dropout,
            "nlayer": k_nlayer,
            "nclass": 1
        }

        # Classifier model hyperparameters
        self.c_hidden = c_hidden
        self.c_input = c_input

        # Graphair model hyperparameters
        self.graphair_hyperparams = {
            "k_lr": k_lr,
            "c_lr": c_lr,
            "g_lr": g_lr,
            "f_lr": f_lr,
            "temperature": graphair_temperature,
            "g_warmup_lr": g_warmup_lr,
            "weight_decay": weight_decay,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "lam": lam
        }

    def initialize_dataset(self, dataset_name):
        if dataset_name == Datasets.NBA:
            return NBA(device=self.device)
        elif dataset_name == Datasets.POKEC_N:
            return POKEC(device=self.device, dataset_sample="pokec_n")
        elif dataset_name == Datasets.POKEC_Z:
            return POKEC(device=self.device, dataset_sample="pokec_z")
        else:
            raise Exception(
                f"Dataset {dataset_name} is not supported. Available datasets are: {[Datasets.POKEC_Z, Datasets.POKEC_N, Datasets.NBA]}"
            )

    def run(self):
        """Runs training and evaluation for a fairgraph model on the given dataset."""

        # Initialize augmentation model g
        self.aug_model = aug_module(
            features=self.dataset.features,
            device=self.device,
            **self.g_hyperparams
        ).to(self.device)

        # Initialize encoder model f
        self.f_encoder = GCN_Body(
            in_feats=self.dataset.features.shape[1],
            **self.f_hyperparams
        ).to(self.device)

        # Initialize adversary model k
        self.sens_model = GCN(
            in_feats=self.dataset.features.shape[1],
            **self.k_hyperparams
        ).to(self.device)

        # Initialize classifier for testing
        self.classifier_model = Classifier(
            input_dim=self.c_input, hidden_dim=self.c_hidden
        )

        # Initialize the Graphair model
        self.model = Graphair(
            aug_model=self.aug_model,
            f_encoder=self.f_encoder,
            sens_model=self.sens_model,
            classifier_model=self.classifier_model,
            dataset=self.dataset.name,
            **self.graphair_hyperparams
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
            adv_epoches=1,
            verbose=self.verbose
        )  # TODO: figure out what adv_epochs is
        print("Training time: ", time.time() - st_time)

        # Test the model
        results = self.model.test(
            adj=self.dataset.adj,
            features=self.dataset.features,
            labels=self.dataset.labels,
            epochs=self.test_epochs,
            idx_train=self.dataset.idx_train,
            idx_val=self.dataset.idx_val,
            idx_test=self.dataset.idx_test,
            sens=self.dataset.sens,
            verbose=self.verbose
        )

        return results
    
    def __repr__(self):
        return f"""Experiment with the following hyperparameters:
        Device: {self.device}, Dataset: {self.dataset.name}, Epochs: {self.epochs}, Test epochs: {self.test_epochs}
        Augmentation model g hyperparameters: {self.g_hyperparams}
        Encoder model f hyperparameters: {self.f_hyperparams}
        Adversary model k hyperparameters: {self.k_hyperparams}"""
