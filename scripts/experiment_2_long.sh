hyperparams_dir="experiments/experiment_2/experiment_2_2"

python run.py \
    --dataset_name POKEC_N \
    --grid_search \
    --n_runs 3 \
    --params_file $hyperparams_dir/hyperparams.yml \
    --experiment_name POKEC_N_grid_search_ours_more_epochs

python run.py \
    --dataset_name POKEC_Z \
    --grid_search \
    --n_runs 3 \
    --params_file $hyperparams_dir/hyperparams.yml \
    --experiment_name POKEC_Z_grid_search_ours_more_epochs
