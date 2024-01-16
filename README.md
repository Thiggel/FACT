# Reproducing Graphair
## Quick start

To run the experiment use:
```
python run.py --params_file <example.yml>  --dataset_name <example> --device <example>
```
`--params_file`: YAML file with hyperparameters, default: hyperparams.yml
`--dataset_name`: dataset to use, default: NBA
`--device`: device to use, default: gpu if available, otherwise cpu

To run with different hyperparameters, either change `hyperparams.yml` or create a new yaml file and pass it as an argument.
