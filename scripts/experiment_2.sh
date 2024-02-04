# run grid search for each dataset
python run.py \
    --dataset_name NBA \
    --grid_search \
    --experiment_name NBA_grid_search_ours

python run.py \
    --dataset_name POKEC_N \
    --grid_search \
    --experiment_name POKEC_N_grid_search_ours

python run.py \
    --dataset_name POKEC_Z \
    --grid_search \
    --experiment_name POKEC_Z_grid_search_ours