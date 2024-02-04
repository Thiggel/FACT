# Reproducing Graphair

This repository includes code for implementations, experiments used for evaluating the reproducibility of the paper ["Learning Fair Graph Representations via Automated Data Augmentations"](https://openreview.net/pdf?id=1_OGWcP1s9w) by Ling et al. (2023).

## Quick start

### Setup
To install the required dependencies:
- Create a python virtual environment: `python -m venv venv`
- Activate the envirionement: `source ./venv/bin/activate`
- Use the latest version of pip: `pip install --upgrade pip`
- Install the required dependencies: `pip install -r requirements.txt`
- Install graphsaint `./install_graphsaint.sh`

### Our experiments
The scripts with the configurations for all the experiments carried our as a part of our reproduction can be found in `/scripts/` directory. To run a script: `. ./scripts/experiment_1.sh`.
Here is a more detailed description of each experiment:

| Experiment Description | File | Notes |
|-------------|------| ----- |
|      Grid search as specified in the original paper       |   ?   |       |
|      Grid search with our evaluation protocol       |   [experiment_2.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_2.sh)   |       |
|      Running with best hyperparameters found in experiment 1 but with our evaluation protocol         |   ?   |       |
|      Disabling adversarial trainin (`alpha` = 0)       |   [experiment_4.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_4.sh)   | |
|      Training a supervised model on the original graph with no augmentations from Graphair       |   [experiment_5.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_5.sh)   | |
|      Ablation study: training Graphair w/o node feature masking an w/o edge perturbation       |   [experiment_6.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_6.sh)   | |
|      Synthetic datasets with different homophily values       |   [experiment_7.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_7.sh)   | |
|      Replacing GCN with GAT       |   [experiment_8.sh](https://github.com/Thiggel/FACT/blob/main/scripts/experiment_8.sh)   | |

### Custom experiments
To run an experiment with custom settings (e.g. different device or hyperparameters):
```
python run.py --params_file hyperparams.yml  --dataset_name NBA --device cpu --verbose --seed 42
```
`--params_file`: YAML file with hyperparameters, default: hyperparams.yml

`--dataset_name`: dataset to use, default: NBA

`--device`: device to use, default: gpu if available, otherwise cpu

`--verbose`: whether to print the training / testing logs

`--seed`: a seed to use for reproducibility

`--grid_search`: whether to run one run, or do a grid search over the hyperparameters

`--grid_hparams`: the values over which to do the hyperparameter search

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
