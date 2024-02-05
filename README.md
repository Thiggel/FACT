# Reproducing Graphair

This repository includes code for implementations, experiments used for evaluating the reproducibility of the paper ["Learning Fair Graph Representations via Automated Data Augmentations"](https://openreview.net/pdf?id=1_OGWcP1s9w) by Ling et al. (2023).

## Setup
To install the required dependencies:
- Create a python virtual environment: `python -m venv venv`
- Activate the envirionement: `source ./venv/bin/activate`
- Use the latest version of pip: `pip install --upgrade pip`
- Install the required dependencies: `pip install -r requirements.txt`
- Install graphsaint `./install-graphsaint.sh` (you might get the following message `error: could not create 'fairgraph/dataset/graphsaint/cython_sampler.*.so': No such file or directory`. Check if `cython_sampler.*.so` file is in `fairgraph/dataset/graphsaint/`. If yes, Graphsaint was compiled)

## Run experiments on NBA dataset
To run all experiments (other than grid searches and training on synthetic datasets) on just the NBA dataset, run the demo:


`. scripts/demo.sh`


If permission to run is denied, run `chmod +x scripts/demo.sh`, then the command above.

## Our experiments
The scripts with the configurations for all the experiments carried our as a part of our reproduction can be found in `/scripts/` directory. To run a script: `. ./scripts/experiment_1.sh`.
Here is a more detailed description of each experiment:

| Experiment Description | File | Notes |
|-------------|------| ----- |
|      Grid search as specified in the original paper       |   [experiment_1.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_1.sh)   |       |
|      Grid search with our evaluation protocol       |   [experiment_2.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_2.sh)   |       |
|      Running with best hyperparameters found in experiment 1 but with our evaluation protocol         |   [experiment_3.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_3.sh)   |   Need to run experiment 2 first as this experiment uses best hyperparams found there    |
|      Disabling adversarial training (`alpha = 0`)       |   [experiment_4.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_4.sh)   | |
|      Training a supervised model on the original graph with no augmentations from Graphair       |   [experiment_5.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_5.sh)   | |
|      Ablation study: training Graphair w/o node feature masking an w/o edge perturbation       |   [experiment_6.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_6.sh)   | |
|      Synthetic datasets with different homophily values       |   [experiment_7.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_7.sh)   | |
|      Replacing GCN with GAT       |   [experiment_8.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_8.sh)   | |

## Custom experiments
To run an experiment with custom settings (e.g. different device or hyperparameters):
```
python run.py --params_file hyperparams.yml  --dataset_name NBA --device cpu --verbose --seed 42
```
`--params_file`: YAML file with hyperparameters, default: hyperparams.yml

`--dataset_name`: dataset to use, default: NBA

`--device`: device to use, default: gpu if available, otherwise cpu

`--verbose`: print the training / testing logs

`--seed`: a seed to use for reproducibility, default: 42

`--grid_search`: whether to run one run, or do a grid search over the hyperparameters

`--grid_hparams`: the values over which to do the hyperparameter search, default: (0.1, 1, 10)

`--n_runs`: number of experiment runs, default: 5

`--n_tests`: number of tests for each experiment, default: 1

`--supervised_testing`: whether to only run supervised testing and skip training Graphair, default: False

To run with different hyperparameters, either change `hyperparams.yml` or create a new yaml file and pass it as an argument.

### To Reproduce Spearman Correlation and Node Sensitive Homophily Figures

```
python reproduce-figures.py --alpha 0.1 --beta 1.0 --gamma 1.0 --lambda 10.0 --dataset NBA
python reproduce-figures.py --alpha 1 --beta 1 --gamma 10 --lambda 0.1 --dataset POKEC_N
python reproduce-figures.py --alpha 1 --beta 1 --gamma 1 --lambda 0.1 --dataset POKEC_Z
```
