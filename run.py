import argparse
import yaml

from fairgraph import Experiment


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--params_file', default='./hyperparams.yml', type=str,
                        help='Path to the file with hyperparameters')
    parser.add_argument('--dataset_name', default='NBA', type=str,
                        help='The dataset to use')
    parser.add_argument('--device', type=str, nargs='?',
                        help='Device to use')
    parser.add_argument('--verbose', action=argparse.BooleanOptionalAction,
                        help='Whether to print the training logs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducibility')
    parser.add_argument('--grid_search', action=argparse.BooleanOptionalAction,
                        help='Whether to run grid seach')
    parser.add_argument('--grid_hparams', type=float, nargs='*',
                        help='Which hyperparameters are used for the grid search')
    parser.add_argument('--hmm', type=float, default=0.8,
                        help='If using synthetic data, the hyperparameters for the hmm')
    parser.add_argument('--hMM', type=float, default=0.2, 
                        help='If using synthetic data, the hyperparameters for the hMM')
    parser.add_argument('--attention', action=argparse.BooleanOptionalAction,
                        help='Whether to use graph attention instead of convolution')
    parser.add_argument('--n_runs', default=5, type=int,
                        help='Number of experiment runs')
    parser.add_argument('--n_tests', default=1, type=int,
                        help='Number of tests for each experiment')
    

    args = parser.parse_args()
    kwargs = vars(args)

    with open(args.params_file, "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    print(args)

    # Initialize and run an experiment
    experiment = Experiment(
        experiment_name=args.experiment_name,
        params_file=args.params_file,
        dataset_name=args.dataset_name,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        synthetic_hmm=args.hmm,
        synthetic_hMM=args.hMM,
        use_graph_attention=args.attention,
        n_runs=args.n_runs,
        n_tests=args.n_tests,
        **hyperparams,
    )

    print(experiment)

    if args.grid_search:
        results = experiment.run_grid_search(args.grid_hparams)
        print('------ best results: ------')
        print(results)
    else: 
        results = experiment.run()
    
    # save to file
    import pickle
    with open('results.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
