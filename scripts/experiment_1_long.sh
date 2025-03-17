hyperparams_dir="experiments/experiment_1/experiment_1_2"

# run grid search for each dataset
python run.py \
    --dataset_name POKEC_N \
    --grid_search \
    --n_runs 1 \
    --n_tests 5 \
    --params_file $hyperparams_dir/hyperparams.yml \
    --experiment_name POKEC_N_grid_search_original_longer_training

python run.py \
    --dataset_name POKEC_Z \
    --grid_search \
    --n_runs 1 \
    --n_tests 5 \
    --params_file $hyperparams_dir/hyperparams.yml \
    --experiment_name POKEC_Z_grid_search_original_longer_training
