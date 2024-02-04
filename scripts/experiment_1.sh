# run grid search for each dataset
python run.py \
    --dataset_name NBA \
    --grid_search \
    --n_runs 1 \
    --n_test 5 \
    --experiment_name NBA_grid_search_original

python run.py \
    --dataset_name POKEC_N \
    --grid_search \
    --n_runs 1 \
    --n_tests 5 \
    --experiment_name POKEC_N_grid_search_original

python run.py \
    --dataset_name POKEC_Z \
    --grid_search \
    --n_runs 1 \
    --n_tests 5 \
    --experiment_name POKEC_Z_grid_search_original