# for this experiment, we run the NBA dataset with the best hyperparameters we found using grid search, but we set alpha to zero
python run.py   \
    --dataset NBA \
    --params_file experiments/experiment_2/nba_best_hyperparameters_alpha_0.yml \
    --experiment_name NBA_alpha_0
