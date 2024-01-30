# Train a supervised model on the original graph with no augmentations from Graphair
python3 run.py --experiment_name "experiment_5_1_NBA" --device cuda --dataset NBA --n_tests 5 --skip_graphair --params_file experiments/experiment_2/nba_best_hyperparameters.yml
python3 run.py --experiment_name "experiment_5_1_POKEC_N" --device cuda --dataset POKEC_N --n_tests 5 --skip_graphair --params_file experiments/experiment_2/pokec_n_best_hyperparameters.yml
python3 run.py --experiment_name "experiment_5_1_POKEC_Z" --device cuda --dataset POKEC_Z --n_tests 5 --skip_graphair --params_file experiments/experiment_2/pokec_z_best_hyperparameters.yml

# Train a GCN classifier on the output of Graphair to be able to compare with the above
python3 run.py --experiment_name "experiment_5_2_NBA" --device cuda --dataset NBA --n_runs 1 --n_tests 5 --use_gcn_classifier --params_file experiments/experiment_2/nba_best_hyperparameters.yml
python3 run.py --experiment_name "experiment_5_2_POKEC_N" --device cuda --dataset POKEC_N --n_runs 1 --n_tests 5 --use_gcn_classifier --params_file experiments/experiment_2/pokec_n_best_hyperparameters.yml
python3 run.py --experiment_name "experiment_5_2_POKEC_Z" --device cuda --dataset POKEC_Z --n_runs 1 --n_tests 5 --use_gcn_classifier --params_file experiments/experiment_2/pokec_z_best_hyperparameters.yml
