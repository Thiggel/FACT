# Reproducing Graphair

## Quick start

To install the required dependencies:
- Create a python virtual environment: `python -m venv venv`
- Activate the envirionement: `source ./venv/bin/activate`
- Install the required dependencies: `pip install -r requirements.txt`

To run the experiment use:
```
python run.py --params_file hyperparams.yml  --dataset_name NBA --device cpu --verbose --seed 42
```
`--params_file`: YAML file with hyperparameters, default: hyperparams.yml

`--dataset_name`: dataset to use, default: NBA

`--device`: device to use, default: gpu if available, otherwise cpu

`--verbose`: whether to print the training / testing logs

`--seed`: a seed to use for reproducibility

To run with different hyperparameters, either change `hyperparams.yml` or create a new yaml file and pass it as an argument.
