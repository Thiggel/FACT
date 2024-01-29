# for this experiment, we run the datasets with the best hyperparameters we found using grid search, but we set alpha to zero
python run.py --dataset NBA --params_file experiments/experiment_2/nba_best_hyperparameters_alpha_0.yml
python run.py --dataset POKEC_Z --params_file experiments/experiment_2/pokec_z_best_hyperparameters_alpha_0.yml
python run.py --dataset POKEC_N --params_file experiments/experiment_2/pokec_n_best_hyperparameters_alpha_0.yml