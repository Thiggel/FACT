import os
import shutil
from torch.utils.tensorboard import SummaryWriter
import time
from .method.Graphair import Graphair, aug_module, GCN, GCN_Body, Classifier, GAT_Body, GAT_Model
from .utils.constants import Datasets
from .utils.utils import set_device, set_seed
from .dataset import POKEC, NBA, ArtificialSensitiveGraphDataset

import time
from enum import Enum
import sys

# TODO: go through all the models and replace hardcoded hyperparemters with arguments, then add to hyperparams file


class Objective:
    class ObjectiveType(Enum):
        ACCURACY = 0
        FAIRNESS = 1

    def __init__(self, objective: ObjectiveType):
        self.objective = objective
        self.best_value = self.get_initial_value()

    def get_initial_value(self):
        return {
            self.ObjectiveType.ACCURACY: -1,
            self.ObjectiveType.FAIRNESS: 999,
        }[self.objective]

    def compare(self, new_val: float) -> bool:
        return {
            self.ObjectiveType.ACCURACY: new_val > self.best_value,
            self.ObjectiveType.FAIRNESS: new_val < self.best_value
        }[self.objective]

    def is_better(self, res_dict: dict) -> bool:
        new_val = self.get_result(res_dict)

        if self.compare(new_val):
            self.best_value = new_val
            return True

        return False

    def get_result(self, res_dict: dict) -> float:
        return {
            self.ObjectiveType.ACCURACY: res_dict['acc']['mean'],
            self.ObjectiveType.FAIRNESS:
                res_dict['dp']['mean'] + res_dict['eo']['mean'],
        }[self.objective]


class Experiment:
    """
    Creates an experiment with the specified hyperparameters. Instantiates
    Graphair model and implements method to train and evaluate.
    """

    def __init__(
        self,
        experiment_name,
        params_file,
        dataset_name,
        device=None,
        verbose=False,
        epochs=10_000,
        test_epochs=1_000,
        batch_size=1000,
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
        graphair_temperature=0.07,
        synthetic_hmm=0.8,
        synthetic_hMM=0.2,
        use_graph_attention=False,
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
        self.name = experiment_name
        self.device = device if device else set_device()
        self.batch_size = batch_size
        self.dataset = self.initialize_dataset(dataset_name, synthetic_hmm, synthetic_hMM)
        self.verbose = verbose

        # Set a seed for reproducibility
        set_seed(seed)
        self.seed = seed

        # Trainig hyperparameters
        self.warmup = warmup
        self.epochs = epochs
        self.test_epochs = test_epochs

        self.use_graph_attention = use_graph_attention

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

        self.params_file = params_file
        self.initialize_logging()

    def initialize_logging(self):
        log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=log_dir)
        shutil.copy(self.params_file, log_dir)
        sys.stdout = open(os.path.join(log_dir, "output.txt"), "w")

    def initialize_dataset(
        self,
        dataset_name,
        synthetic_hmm=0.8,
        synthetic_hMM=0.2
    ):
        if dataset_name == Datasets.NBA:
            return NBA(device=self.device)
        elif dataset_name == Datasets.POKEC_N:
            return POKEC(device=self.device, dataset_sample="pokec_n", batch_size=self.batch_size)
        elif dataset_name == Datasets.POKEC_Z:
            return POKEC(device=self.device, dataset_sample="pokec_z", batch_size=self.batch_size)
        elif dataset_name == Datasets.SYNTHETIC:
            print(synthetic_hmm, synthetic_hMM)
            return ArtificialSensitiveGraphDataset(
                path=os.getcwd() + '/fairgraph/dataset/dataset/artificial/' +
                'DPAH-N1000-fm0.3-d0.03-ploM2.5-plom2.5-' +
                f'hMM{synthetic_hMM}-hmm{synthetic_hmm}-ID0.gpickle',
                device=self.device
            )
        else:
            raise Exception(
                f"Dataset {dataset_name} is not supported. Available datasets are: {[Datasets.POKEC_Z, Datasets.POKEC_N, Datasets.NBA]}"
            )
        
    def run_grid_search(
        self,
        hparam_values,
        objective: Objective.ObjectiveType = Objective.ObjectiveType.FAIRNESS
    ):
        """
        Runs grid seach using the given hyperparameter values

        Args:
            hparam_values (tuple): the values alpha, gamma 
                and lam can take in the grid search.
            objective (GridSearchObjective): whether the grid search
                should optimize fairness or accuracy

        Returns:
            best_params (dict): values of the hyperparameters
                for the setting with the best accuracy.
            best_res_dict (dict): output of self.run for the
                best hyperparameter values.
        """
        objective = Objective(objective)

        hparam_values = hparam_values if hparam_values else (0.1, 1., 10.)

        best_params = None
        best_res_dict = None

        beta = 1.
        for alpha in hparam_values:
            for gamma in hparam_values:
                for lam in hparam_values:
                    self.graphair_hyperparams['alpha'] = alpha
                    self.graphair_hyperparams['beta'] = beta
                    self.graphair_hyperparams['gamma'] = gamma
                    self.graphair_hyperparams['lam'] = lam

                    print(f"alpha: {self.graphair_hyperparams['alpha']}, " +
                          f"lambda: {self.graphair_hyperparams['lam']}, " +
                          f"gamma: {self.graphair_hyperparams['gamma']}")

                    res_dict = self.run()

                    if objective.is_better(res_dict):
                        best_res_dict = res_dict

                        best_params = {
                            'alpha': alpha,
                            'beta': beta,
                            'gamma': gamma,
                            'lam': lam
                        }

        return best_params, best_res_dict

    def run(self):
        """Runs training and evaluation for a fairgraph model on the given dataset."""
        
        # Set the random seed
        set_seed(self.seed)

        # Initialize augmentation model g
        self.aug_model = aug_module(
            features=self.dataset.features,
            device=self.device,
            use_graph_attention=self.use_graph_attention,
            **self.g_hyperparams
        ).to(self.device)

        # Initialize encoder model f
        self.f_encoder = GCN_Body(
            in_feats=self.dataset.features.shape[1],
            **self.f_hyperparams
        ).to(self.device) if not self.use_graph_attention else GAT_Body(
            in_feats=self.dataset.features.shape[1],
            **self.f_hyperparams
        ).to(self.device)

        # Initialize adversary model k
        self.sens_model = GCN(
            in_feats=self.dataset.features.shape[1],
            **self.k_hyperparams
        ).to(self.device) if not self.use_graph_attention else GAT_Model(
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
            device=self.device,
            dataset=self.dataset.name,
            **self.graphair_hyperparams
        ).to(self.device)

        # Train the model
        print("Start training")
        if self.dataset.name in [Datasets.POKEC_Z, Datasets.POKEC_N]:
            # call fit_batch_GraphSAINT
            st_time = time.time()
            self.model.fit_batch_GraphSAINT(
                epochs=self.epochs,
                adj=self.dataset.adj,
                x=self.dataset.features,
                sens=self.dataset.sens,
                idx_sens=self.dataset.idx_sens_train,
                minibatch=self.dataset.minibatch,
                warmup=self.warmup,
                adv_epoches=1,
                verbose=self.verbose,
                writer=self.writer,
                )
            print("Training time: ", time.time() - st_time)

        else:
            st_time = time.time()
            self.model.fit_whole(
                epochs=self.epochs,
                adj=self.dataset.adj,
                x=self.dataset.features,
                sens=self.dataset.sens,
                idx_sens=self.dataset.idx_sens_train,
                warmup=self.warmup,
                adv_epoches=1,
                verbose=self.verbose,
                writer=self.writer,
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
            verbose=self.verbose,
            writer=self.writer,
        )

        return results

    def create_log_dir(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("experiments", f"{timestamp}_{self.name}")
        os.makedirs(log_dir, exist_ok=True)

        return log_dir
    
    def __repr__(self):
        return f"""Experiment with the following hyperparameters:
        Device: {self.device}, Dataset: {self.dataset.name}, Epochs: {self.epochs}, Test epochs: {self.test_epochs}, Batch size: {self.batch_size}
        Augmentation model g hyperparameters: {self.g_hyperparams}
        Encoder model f hyperparameters: {self.f_hyperparams}
        Adversary model k hyperparameters: {self.k_hyperparams}"""
