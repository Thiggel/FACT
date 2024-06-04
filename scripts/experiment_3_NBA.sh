hyperparams_dir="experiments/experiment_2"

# train a model with the best hyperparameters once for the NBA dataset
python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --dataset NBA \
    --n_runs 5 \
    --experiment_name NBA_best_hparams
