import os
import shutil
import sys
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import NBA, POKEC, ArtificialSensitiveGraphDataset
from .method.Graphair import (
    GCN,
    Classifier,
    GAT_Body,
    GAT_Model,
    GCN_Body,
    Graphair,
    aug_module
)
from .utils.constants import Datasets
from .utils.utils import (
    set_device,
    set_seed,
    find_pareto_front,
    plot_pareto,
    get_grid_search_results_from_dir,
)


class Logger(object):
    def __init__(self, log_file_name):
        """log both to a file and the terminal"""
        self.terminal = sys.stdout
        self.log_file = open(log_file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        """needed for Python 3 compatibility"""
        pass


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
        edge_perturbation=True,
        node_feature_masking=True,        
        synthetic_hmm=0.8,
        synthetic_hMM=0.2,
        use_graph_attention=False,
        n_runs=5,
        n_tests=1,
        grid_search_resume_dir=None,
        skip_graphair=False,
    ):
        """
        Initializes an Experiment class instance.

        Args:
            experiment_name (str): name of the experiment.
            params_file (str): path to the file containing the hyperparameters.
            dataset_name (str): name of the dataset to use.
            device (str): device to use for training and evaluation.
            verbose (bool): whether to print training progress. Defaults to False.
            epochs (int): number of epochs to train for. Defaults to 10000.
            test_epochs (int): number of epochs to test for. Defaults to 1000.
            batch_size (int): batch size to use for training. Defaults to 1000.
            seed (int): random seed to use for reproducibility. Defaults to 42.
            weight_decay (float): weight decay to use for training. Defaults to 1e-5.
            g_temperature (float): temperature to use for the augmentation model. Defaults to 1.0.
            g_hidden (int): number of hidden units to use for the augmentation model. Defaults to 64.
            g_dropout (float): dropout to use for the augmentation model. Defaults to 0.1.
            g_nlayer (int): number of layers to use for the augmentation model. Defaults to 1.
            mlpx_dropout (float): dropout to use for the MLPX module. Defaults to 0.1.
            f_hidden (int): number of hidden units to use for the encoder model. Defaults to 64.
            f_layers (int): number of layers to use for the encoder model. Defaults to 2.
            f_dropout (float): dropout to use for the encoder model. Defaults to 0.1.
            f_output_features (int): number of output features to use for the encoder model. Defaults to 64.
            k_hidden (int): number of hidden units to use for the adversary model. Defaults to 64.
            k_output_features (int): number of output features to use for the adversary model. Defaults to 64.
            k_dropout (float): dropout to use for the adversary model. Defaults to 0.1.
            k_nlayer (int): number of layers to use for the adversary model. Defaults to 2.
            c_hidden (int): number of hidden units to use for the classifier model. Defaults to 64.
            c_input (int): number of input features to use for the classifier model. Defaults to 64.
            warmup (int): number of warmup epochs to use for training the adversary model. Defaults to 0.
            alpha (float): weight of the adversarial loss. Defaults to 20.
            beta (float): weight of the contrastive loss. Defaults to 0.9.
            gamma (float): weight of the reconstruction loss. Defaults to 0.7.
            lam (float): weight of the node reconstruction loss term. Defaults to 1.
            k_lr (float): learning rate to use for the adversary model. Defaults to 1e-4.
            c_lr (float): learning rate to use for the classifier model. Defaults to 1e-3.
            g_lr (float): learning rate to use for the augmentation model. Defaults to 1e-4.
            g_warmup_lr (float): learning rate to use for the augmentation model during warmup. Defaults to 1e-3.
            f_lr (float): learning rate to use for the encoder model. Defaults to 1e-4.
            graphair_temperature (float): temperature to use for the Graphair model. Defaults to 0.07.
            edge_perturbation (bool): whether to use edge perturbation. Defaults to True.
            node_feature_masking (bool): whether to use node feature masking. Defaults to True.
            synthetic_hmm (float): probability of a node being connected to a node of the same class in the synthetic dataset. Defaults to 0.8.
            synthetic_hMM (float): probability of a node being connected to a node of a different class in the synthetic dataset. Defaults to 0.2.
            use_graph_attention (bool): whether to use graph attention instead of GCN. Defaults to False.
            n_runs (int): number of training runs to average over. Defaults to 5.
            n_tests (int): number of tests to average over after each training run. Defaults to 1.
            grid_search_resume_dir (str): path to the directory containing the results of a grid search to resume. Defaults to None.
            skip_graphair (bool): whether to skip training the Graphair model and evaluate by directly training the GNN encoder + MLP. Defaults to False.
        """
        self.name = experiment_name
        
        if device in ["cpu", "cuda", "mps"]:
            self.device = torch.device(device)
        else:
            self.device = set_device()
        
        self.batch_size = batch_size
        self.dataset = self.initialize_dataset(dataset_name, synthetic_hmm, synthetic_hMM)
        self.verbose = verbose
        self.n_runs = n_runs
        self.n_tests = n_tests
        self.skip_graphair = skip_graphair

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
            "edge_perturbation": edge_perturbation,
            "node_feature_masking": node_feature_masking,
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

        self.grid_search_resume_dir = grid_search_resume_dir

        self.params_file = params_file
        self.initialize_logging()

    def initialize_logging(self):
        self.log_dir = self.create_log_dir()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        shutil.copy(self.params_file, self.log_dir)
        self.logger = Logger(os.path.join(self.log_dir, "output.txt"))
        sys.stdout = self.logger

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
                f"Dataset {dataset_name} is not supported. Available datasets are: {[Datasets.POKEC_Z, Datasets.POKEC_N, Datasets.NBA, Datasets.SYNTHETIC]}"
            )

    def run_grid_search(
        self,
        hparam_values,
    ):
        """
        Runs grid seach using the given hyperparameter values

        Args:
            hparam_values (tuple): the values alpha, gamma
                and lam can take in the grid search.

        Returns:
            best_params (dict): values of the hyperparameters
                for the setting with the best accuracy.
            best_res_dict (dict): output of self.run for the
                best hyperparameter values.
        """
        hparam_values = hparam_values if hparam_values else (0.1, 1., 10.)

        beta = 1.

        results = []

        if self.grid_search_resume_dir is not None:
            results, finished_hparams = get_grid_search_results_from_dir(self.grid_search_resume_dir)
            for hparams in finished_hparams:
                [a, g, l] = hparams
                shutil.copy(
                    os.path.join(self.grid_search_resume_dir, f"output-a{a}-b{beta}-g{g}-l{l}.txt"),
                    self.log_dir,
                )

        for alpha in hparam_values:
            for gamma in hparam_values:
                for lam in hparam_values:
                    if self.grid_search_resume_dir is not None and [alpha, gamma, lam] in finished_hparams:
                        continue

                    self.logger.log_file.close()
                    self.logger.log_file = open(os.path.join(self.log_dir, f"output-a{alpha}-b{beta}-g{gamma}-l{lam}.txt"), "w")

                    self.graphair_hyperparams['alpha'] = alpha
                    self.graphair_hyperparams['beta'] = beta
                    self.graphair_hyperparams['gamma'] = gamma
                    self.graphair_hyperparams['lam'] = lam

                    print(f"alpha: {self.graphair_hyperparams['alpha']}, " +
                          f"lambda: {self.graphair_hyperparams['lam']}, " +
                          f"gamma: {self.graphair_hyperparams['gamma']}")

                    res_dict = self.run()

                    results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'lam': lam,
                        'acc': {
                            'mean': res_dict['acc']['mean'],
                            'std': res_dict['acc']['std'],
                        },
                        'dp': {
                            'mean': res_dict['dp']['mean'],
                            'std': res_dict['dp']['std'],
                        },
                        'eo': {
                            'mean': res_dict['eo']['mean'],
                            'std': res_dict['eo']['std'],
                        },
                    })

        best_accuracy_params = max(
            results, key=lambda x: x['acc']['mean']
        )
        best_dp_params = min(results, key=lambda x: x['dp']['mean']) # Best DP is lowest DP
        best_eo_params = min(results, key=lambda x: x['eo']['mean']) # Best EO is lowest EO

        pareto_front_dp = find_pareto_front(results, metric1='acc', metric2='dp')
        pareto_front_eo = find_pareto_front(results, metric1='acc', metric2='eo')

        print('Grid Search Results:\n' +
              'Best Accuracy: ' + str(best_accuracy_params) + '\n' +
              'Best DP: ' + str(best_dp_params) + '\n' +
              'Best EO: ' + str(best_eo_params) + '\n')

        print('Pareto Front (Accuracy - DP):\n', pareto_front_dp)
        print('Pareto Front (Accuracy - EO):\n', pareto_front_eo)

        print('All Results:\n', results)

        for fairness_metric in ['dp', 'eo']:
            for show_all in [True, False]:
                plot_pareto(
                    results=results,
                    fairness_metric=fairness_metric,
                    show_all=show_all,
                    dataset=self.dataset.name,
                    filepath=os.path.join(self.log_dir, f"{fairness_metric.upper()}-Acc_pareto{'_all' if show_all else ''}.svg"),
                )

        self.logger.log_file.close()
        self.logger.log_file = open(os.path.join(self.log_dir, "output.txt"), "a")

    def run(self):
        """Runs training and evaluation for a fairgraph model on the given dataset."""

        print("Start training")
        training_times, accuracies, dps, eos = [], [], [], []

        for i in range(self.n_runs):

            # Set the random seed
            set_seed(self.seed + i)

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
                n_tests=self.n_tests,
                skip_graphair=self.skip_graphair,
                **self.graphair_hyperparams
            ).to(self.device)

            if not self.skip_graphair:
                start_time = time.time()
                if self.dataset.name in [Datasets.POKEC_Z, Datasets.POKEC_N]:
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
                else:
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
                    )

                training_time = time.time() - start_time
                print(f"Training time: {training_time:.2f}")
                training_times.append(training_time)

                avg_time_per_epoch = training_time / self.epochs
                print(f"Average time per epoch: {avg_time_per_epoch:.4f}")
            else:
                print("Skipping training")

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

            if self.skip_graphair:
                print(results)
                return results

            # Collect the results
            accuracies.append(results['acc']['mean'])
            eos.append(results['eo']['mean'])
            dps.append(results['dp']['mean'])
            print(f"Run {i} results: {results}")

        average_results = {
            "acc": {"mean": np.mean(accuracies), "std": np.std(accuracies)},
            "dp": {"mean": np.mean(dps), "std": np.std(dps)},
            "eo": {"mean": np.mean(eos), "std": np.std(eos)}
        }
        
        print(f"Average training time per run: {np.mean(training_times):.2f}")
        print(f"Average results: {average_results}")

        return average_results

    def create_log_dir(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join("experiments", f"{timestamp}_{self.name}")
        os.makedirs(log_dir, exist_ok=True)

        return log_dir
    
    def __repr__(self):
        return f"""Experiment with the following hyperparameters:
        Dataset: {self.dataset.name}, Epochs: {self.epochs}, Test epochs: {self.test_epochs}, Batch size: {self.batch_size}
        Device: {self.device}, Seed: {self.seed}, N Runs: {self.n_runs}, N Tests: {self.n_tests}
        Augmentation model g hyperparameters: {self.g_hyperparams}
        Encoder model f hyperparameters: {self.f_hyperparams}
        Adversary model k hyperparameters: {self.k_hyperparams}"""
