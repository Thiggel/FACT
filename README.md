# Reproducing Graphair

## Quick start

To install the required dependencies:
- Create a python virtual environment: `python -m venv venv`
- Activate the envirionement: `source ./venv/bin/activate`
- Use the latest version of pip: `pip install --upgrade pip`
- Install the required dependencies: `pip install -r requirements.txt`
- Install graphsaint `./install_graphsaint.sh`

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

To run with different hyperparameters, either change `hyperparams.yml` or create a new yaml file and pass it as an argument.
