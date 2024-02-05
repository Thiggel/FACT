hyperparams_dir="experiments/experiment_6"

# Without node feature masking
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_fm_nba.yml \
    --dataset_name NBA \
    --experiment_name NBA_without_fm

# Without edge perturbation
python run.py \
    --params_file $hyperparams_dir/hyperparams_no_ep_nba.yml \
    --dataset_name NBA \
    --experiment_name NBA_without_ep
