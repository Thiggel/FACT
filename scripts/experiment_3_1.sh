hyperparams_dir="experiments/experiment_3/experiment_3_1"

# train a model with the best hyperparameters once for each dataset
    
python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --dataset_name POKEC_N \
    --n_runs 5 \
    --experiment_name POKEC_N_best_hparams_more_epochs

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --dataset POKEC_Z \
    --n_runs 5 \
    --experiment_name POKEC_Z_best_hparams_more_epochs
