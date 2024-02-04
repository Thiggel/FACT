# Train a supervised model on the original graph with no augmentations from Graphair
python3 run.py \
    --dataset_name NBA \
    --n_tests 5 \
    --skip_graphair \
    --params_file experiments/experiment_2/nba_best_hyperparameters.yml \
    --experiment_name NBA_supervised \

python3 run.py \
    --dataset_name POKEC_N \
    --n_tests 5 \
    --skip_graphair \
    --params_file experiments/experiment_2/pokec_n_best_hyperparameters.yml \
    --experiment_name POKEC_N_supervised

python3 run.py \
    --dataset_name POKEC_Z \
    --n_tests 5 \
    --skip_graphair \
    --params_file experiments/experiment_2/pokec_z_best_hyperparameters.yml \
    --experiment_name POKEC_Z_supervised
