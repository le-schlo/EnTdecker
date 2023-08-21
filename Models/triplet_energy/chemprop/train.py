import chemprop
import torch

import sys

arguments = [
    '--data_path', '/Data/EnTdecker_data.csv',
    '--dataset_type', 'regression',
    '--save_dir', '/Models/triplet_energy/CatBoost/chemprop/',
    '--features_generator', 'rdkit_2d_normalized',
    '--no_features_scaling',
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)