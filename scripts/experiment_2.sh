hyperparams_dir="experiments/experiment_2"

# train a model with the best hyperparameters once for each dataset

python run.py \
    --params_file $hyperparams_dir/nba_best_hyperparameters.yml \
    --dataset NBA \
    --experiment_name best_model_training_nba \
    --n_runs 1
    
python run.py \
    --params_file $hyperparams_dir/pokec_n_best_hyperparameters.yml \
    --dataset_name POKEC_N \
    --experiment_name best_model_training_pokec_n \
    --n_runs 1

python run.py \
    --params_file $hyperparams_dir/pokec_z_best_hyperparameters.yml \
    --dataset POKEC_Z \
    --experiment_name best_model_training_pokec_z \
    --n_runs 1
