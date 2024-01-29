# Train a supervised model on the original graph with no augmentations from Graphair
python3 run.py --device cuda --dataset NBA --n_tests 5 --skip_graphair --use_gcn_classifier --params_file hyperparams/experiment_5.yml
python3 run.py --device cuda --dataset POKEC_N --n_tests 5 --skip_graphair --use_gcn_classifier --params_file hyperparams/experiment_5.yml
python3 run.py --device cuda --dataset POKEC_Z --n_tests 5 --skip_graphair --use_gcn_classifier --params_file hyperparams/experiment_5.yml

# Train a GCN classifier on the output of Graphair to be able to compare with the above
python3 run.py --device cuda --dataset NBA --n_tests 5 --use_gcn_classifier --params_file hyperparams/experiment_5.yml
python3 run.py --device cuda --dataset POKEC_N --n_tests 5 --use_gcn_classifier --params_file hyperparams/experiment_5.yml
python3 run.py --device cuda --dataset POKEC_Z --n_tests 5 --use_gcn_classifier --params_file hyperparams/experiment_5.yml
