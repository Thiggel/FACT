# Without node feature masking
python run.py --params_file experiments/hyperparams_no_fm_nba.yml --device cuda --dataset NBA --experiment_name witout_fm_NBA
python run.py --params_file experiments/hyperparams_no_fm_pokec_n.yml --device cuda --dataset POKEC_N --experiment_name witout_fm_POKEC_N
python run.py --params_file experiments/hyperparams_no_fm_pokec_z.yml --device cuda --dataset POKEC_Z --experiment_name witout_fm_POKEC_Z

# Without edge perturbation
python run.py --params_file experiments/hyperparams_no_ep_nba.yml --device cuda --dataset NBA --experiment_name witout_ep_NBA
python run.py --params_file experiments/hyperparams_no_ep_pokec_n.yml --device cuda --dataset POKEC_N --experiment_name witout_ep_POKEC_N
python run.py --params_file experiments/hyperparams_no_ep_pokec_z.yml --device cuda --dataset POKEC_Z --experiment_name witout_ep_POKEC_Z

