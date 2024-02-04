hyperparams_dir="experiments/experiment_6"

# Without node feature masking
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_nba.yml \
    --dataset_name NBA \
    --experiment_name NBA_without_fm
    
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_pokec_n.yml \
    --dataset_name POKEC_N \
    --experiment_name POKEC_N_without_fm

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_pokec_z.yml \
    --dataset_name POKEC_Z \
    --experiment_name POKEC_Z_without_fm

# Without edge perturbation
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_nba.yml \
    --dataset_name NBA \
    --experiment_name NBA_without_ep

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_pokec_n.yml \
    --dataset_name POKEC_N \
    --experiment_name POKEC_N_without_ep

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_pokec_z.yml \
    --dataset_name POKEC_Z \
    --experiment_name POKEC_Z_without_ep
