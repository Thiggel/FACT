# Train a supervised model on the original graph with no augmentations from Graphair
python3 run.py \
    --dataset_name NBA \
    --n_tests 5 \
    --skip_graphair \
    --params_file experiments/experiment_2/nba_best_hyperparameters.yml \
    --experiment_name NBA_supervised \
