# Reproducing Graphair

## Quick start

### Setup
To install the required dependencies:
- Create a python virtual environment: `python -m venv venv`
- Activate the envirionement: `source ./venv/bin/activate`
- Use the latest version of pip: `pip install --upgrade pip`
- Install the required dependencies: `pip install -r requirements.txt`
- Install graphsaint `./install_graphsaint.sh`

### Experiments
To run the experiment use:
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
cd fairgraph
python reproduce-figures.py --alpha 0.1 --beta 1.0 --gamma 1.0 --lambda 10.0 --dataset NBA
python reproduce-figures.py --alpha 1.0 --beta 1.0 --gamma 10.0 --lambda 0.1 --dataset POKEC-Z
python reproduce-figures.py --alpha 1.0 --beta 1.0 --gamma 1.0 --lambda 0.1 --dataset POKEC-N
```
