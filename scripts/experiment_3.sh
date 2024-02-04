hyperparams_dir="experiments/experiment_2"

# train a model with the best hyperparameters once for each dataset
python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --dataset NBA \
    --n_runs 1 \
    --experiment_name NBA_best_hparams
    
python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --dataset_name POKEC_N \
    --n_runs 1 \
    --experiment_name POKEC_N_best_hparams

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --dataset POKEC_Z \
    --n_runs 1 \
    --experiment_name POKEC_Z_best_hparams
