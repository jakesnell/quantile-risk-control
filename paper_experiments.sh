# Section 5.1
python scripts/multi_experiments.py balanced_accuracy --dataset=coco --num_trials=1000 --save_csv;
python scripts/multi_experiments.py balanced_accuracy --dataset=go_emotions --num_trials=1000 --save_csv;
python scripts/multi_experiments.py balanced_accuracy --dataset=clip_cifar_100 --num_trials=1000 --save_csv;
# Section 5.2
python scripts/interval_experiments.py balanced_accuracy --dataset=clip_cifar_100 --beta_lo=0.6 --beta_hi=0.9 --grid_size=50  --fixed_pred --num_trials=1000 --save_csv 
python scripts/interval_experiments.py balanced_accuracy --dataset=go_emotions --beta_lo=0.6 --beta_hi=0.9 --grid_size=100 --fixed_pred --num_trials=1000 --save_csv 
python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.85 --beta_hi=0.95 --grid_size=10 --fixed_pred --num_trials=1000 --save_csv 
python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.9 --beta_hi=1.0 --grid_size=50 --fixed_pred --num_trials=1000 --save_csv 
# Fairness
python scripts/fair_interval_experiments.py custom_loss --dataset=nursery --beta_lo=0.9 --beta_hi=1.0 --grid_size=50 --fixed_pred --num_trials=1000 --save_csv
# Section 5.3
python scripts/mean_experiments.py balanced_accuracy --dataset=coco --num_trials=1000 --save_csv
python scripts/var_experiments.py balanced_accuracy --dataset=coco --num_trials=1000 --save_csv
python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.85 --beta_hi=0.95 --grid_size=10 --num_trials=1000 --save_csv
python scripts/interval_experiments.py balanced_accuracy --dataset=coco --beta_lo=0.9 --beta_hi=1.0 --grid_size=50 --num_trials=1000  --save_csv

