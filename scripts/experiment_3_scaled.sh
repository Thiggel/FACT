hyperparams_dir="experiments/experiment_3_scaled"

python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --dataset_name POKEC_N \
    --n_runs 5 \
    --experiment_name exp_3_scaled_POKEC_N_best_hparams

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --dataset POKEC_Z \
    --n_runs 5 \
    --experiment_name exp_3_scaled_POKEC_Z_best_hparams
