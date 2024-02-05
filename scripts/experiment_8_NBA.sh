hyperparams_dir="experiments/experiment_2"

python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --dataset NBA \
    --attention \
    --experiment_name NBA_GAT
