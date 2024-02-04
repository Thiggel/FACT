hyperparams_dir="experiments/experiment_6"

# Without node feature masking
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_nba.yml \
    --dataset NBA \
    --experiment_name witout_fm_NBA
    
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_pokec_n.yml \
    --dataset_name POKEC_N \
    --experiment_name witout_fm_POKEC_N

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_pokec_z.yml \
    --dataset POKEC_Z \
    --experiment_name witout_fm_POKEC_Z

# Without edge perturbation
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_nba.yml \
    --dataset NBA \
    --experiment_name witout_ep_NBA

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_pokec_n.yml \
    --dataset POKEC_N \
    --experiment_name witout_ep_POKEC_N

python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_pokec_z.yml \
    --dataset POKEC_Z \
    --experiment_name witout_ep_POKEC_Z
