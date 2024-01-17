import argparse
import yaml

from fairgraph import Experiment


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

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
    

    args = parser.parse_args()
    kwargs = vars(args)

    with open(args.params_file, "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    print(args)
    # Initialize and run an experiment
    experiment = Experiment(dataset_name=args.dataset_name, seed=args.seed, device=args.device, verbose=args.verbose, **hyperparams)
    print(experiment)
    if args.grid_search:
        results = experiment.run_grid_search(args.grid_hparams)
    else: 
        results = experiment.run()
    