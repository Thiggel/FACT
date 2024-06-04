hyperparams_dir="experiments/experiment_2"

python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --dataset NBA \
    --attention \
    --experiment_name NBA_GAT_GS \
    --grid_search 
    
python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --dataset_name POKEC_N \
    --attention \
    --experiment_name POKEC_N_GAT_GS \
    --grid_search 

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --dataset POKEC_Z \
    --attention \
    --experiment_name POKEC_Z_GAT_GS \
    --grid_search
