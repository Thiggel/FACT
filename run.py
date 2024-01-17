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

    args = parser.parse_args()
    kwargs = vars(args)

    with open(args.params_file, "r") as stream:
        try:
            hyperparams = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    print(args)
    # Initialize and run an experiment
    experiment = Experiment(dataset_name=args.dataset_name, device=args.device, **hyperparams)
    results = experiment.run()
    print(results)