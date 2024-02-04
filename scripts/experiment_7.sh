python3 run.py \
    --dataset_name SYNTHETIC \
    --hMM 0.2 \
    --hmm 0.2 \
    --grid_search \
    --experiment_name SYNTHETIC_grid_search_20_20

python3 run.py \
    --dataset_name SYNTHETIC \
    --hMM 0.2 \
    --hmm 0.8 \
    --grid_search \
    --experiment_name SYNTHETIC_grid_search_20_80

python3 run.py \
    --dataset_name SYNTHETIC \
    --hMM 0.5 \
    --hmm 0.5 \
    --grid_search \
    --experiment_name SYNTHETIC_grid_search_50_50

python3 run.py \
    --dataset_name SYNTHETIC \
    --hMM 0.8 \
    --hmm 0.2 \
    --grid_search \
    --experiment_name SYNTHETIC_grid_search_80_20

python3 run.py \
    --dataset_name SYNTHETIC \
    --hMM 0.8 \
    --hmm 0.8 \
    --grid_search \
    --experiment_name SYNTHETIC_grid_search_80_80
