hyperparams_dir="experiments/experiment_2"

python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --device cuda \
    --dataset NBA \
    --experiment_name experiment_8_GAT_nba \
    --attention
    
python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --device cuda \
    --dataset_name POKEC_N \
    --experiment_name experiment_8_GAT_pokec_n \
    --attention

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --device cuda \
    --dataset POKEC_Z \
    --experiment_name experiment_8_GAT_pokec_z \
    --attention

