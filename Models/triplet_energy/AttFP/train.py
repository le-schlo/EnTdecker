import os
import sys
sys.setrecursionlimit(50000)
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.backends.cudnn.benchmark = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
# from tensorboardX import SummaryWriter
torch.nn.Module.dump_patches = True

import numpy as np
import pandas as pd


from rdkit import Chem

from AttentiveFP.code.AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array, \
    moltosvg_highlight

def train(model, dataset, optimizer, loss_function):
    model.train()
    np.random.seed(epoch)
    valList = np.arange(0, dataset.shape[0])
    # shuffle them
    np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)

    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))

        optimizer.zero_grad()
        loss = 0.0
        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i]
            y_val = batch_df[task + "_normalized"].values
            loss += loss_function(y_pred, torch.Tensor(y_val).squeeze()) * ratio_list[i] ** 2
        loss.backward()
        optimizer.step()


def eval(model, dataset):
    model.eval()
    eval_MAE_list = {}
    eval_MSE_list = {}
    y_val_list = {}
    y_pred_list = {}
    for i, task in enumerate(tasks):
        y_pred_list[task] = np.array([])
        y_val_list[task] = np.array([])
        eval_MAE_list[task] = np.array([])
        eval_MSE_list[task] = np.array([])

    valList = np.arange(0, dataset.shape[0])
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i + batch_size]
        batch_list.append(batch)
    for counter, batch in enumerate(batch_list):
        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.cano_smiles.values

        batch_df = dataset.loc[batch, :]
        smiles_list = batch_df.cano_smiles.values

        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,
                                                                                                     feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom), torch.Tensor(x_bonds),
                                                 torch.cuda.LongTensor(x_atom_index),
                                                 torch.cuda.LongTensor(x_bond_index), torch.Tensor(x_mask))

        for i, task in enumerate(tasks):
            y_pred = mol_prediction[:, i]
            y_val = batch_df[task + "_normalized"].values

            MAE = F.l1_loss(y_pred, torch.Tensor(y_val).squeeze(), reduction='none')
            MSE = F.mse_loss(y_pred, torch.Tensor(y_val).squeeze(), reduction='none')
            y_pred_list[task] = np.concatenate([y_pred_list[task], y_pred.cpu().detach().numpy()])
            y_val_list[task] = np.concatenate([y_val_list[task], y_val])
            eval_MAE_list[task] = np.concatenate([eval_MAE_list[task], MAE.data.squeeze().cpu().numpy()])
            eval_MSE_list[task] = np.concatenate([eval_MSE_list[task], MSE.data.squeeze().cpu().numpy()])

    eval_MAE_normalized = np.array([eval_MAE_list[task].mean() for i, task in enumerate(tasks)])
    eval_MAE = np.multiply(eval_MAE_normalized, np.array(std_list))
    eval_MSE_normalized = np.array([eval_MSE_list[task].mean() for i, task in enumerate(tasks)])
    eval_MSE = np.multiply(eval_MSE_normalized, np.array(std_list))

    return eval_MAE_normalized, eval_MAE, eval_MSE_normalized, eval_MSE

# ----------------------------------- Data Preprocessing -----------------------
task_name = 'EnTdecker'
tasks = [
    "e_t"
]
raw_filename = "Retrain_Disulfide.csv"
feature_filename = raw_filename.replace('.csv', '.pickle')
filename = raw_filename.replace('.csv', '')
prefix_filename = raw_filename.split('/')[-1].replace('.csv', '')
smiles_tasks_df = pd.read_csv('Models/triplet_energy/AttFP/' + raw_filename)

smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ", len(smilesList))
atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []
for smiles in smilesList:
    try:
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print(smiles)
        pass

print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]
smiles_tasks_df['cano_smiles'] = canonical_smiles_list

random_seed = 42
start_time = str(time.ctime()).replace(':', '-').replace(' ', '_')

per_task_output_units_num = 1  # for regression model
output_units_num = len(tasks) * per_task_output_units_num

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb"))
else:
    feature_dicts = save_smiles_dicts(smilesList, filename)

remained_df = smiles_tasks_df[smiles_tasks_df["cano_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
uncovered_df = smiles_tasks_df.drop(remained_df.index)

train_df = remained_df.sample(frac=0.8, random_state=random_seed)
testandvalid = remained_df.drop(train_df.index)
test_df = testandvalid.sample(frac=0.5, random_state=random_seed)
valid_df = testandvalid.drop(test_df.index)

train_df = train_df.reset_index(drop=True)
valid_df = valid_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

training_data = pd.concat([train_df, valid_df])
# get the stats of the seen dataset (the training data)
# which will be used to noramlize the dataset.
columns = ['Task', 'Mean', 'Standard deviation', 'Mean absolute deviation', 'ratio']
mean_list = []
std_list = []
mad_list = []
ratio_list = []
for task in tasks:
    mean = training_data[task].mean()
    mean_list.append(mean)
    std = training_data[task].std()
    std_list.append(std)
    mad = training_data[task].mad()
    mad_list.append(mad)
    ratio_list.append(std / mad)
    train_df[task + '_normalized'] = (train_df[task] - mean) / std
    valid_df[task + '_normalized'] = (valid_df[task] - mean) / std
    test_df[task + '_normalized'] = (test_df[task] - mean) / std


# ----------------------------------- Training --------------------------------------

p_dropout = 0.2
fingerprint_dim = 512
weight_decay = 5.0  # also known as l2_regularization_lambda
learning_rate = 4.0
radius = 2
T = 2
batch_size = 500
epochs = 400

x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
    [canonical_smiles_list[8]], feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

loss_function = nn.MSELoss()
model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout)
model.cuda()
optimizer = optim.Adam(model.parameters(), 10 ** -learning_rate, weight_decay=10 ** -weight_decay)


model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

best_param = {}
best_param["train_epoch"] = 0
best_param["valid_epoch"] = 0
best_param["train_MSE_normalized"] = 9e8
best_param["valid_MSE_normalized"] = 9e8
for epoch in range(epochs):
    train(model, train_df, optimizer, loss_function)
    train_MAE_normalized, train_MAE, train_MSE_normalized, train_MSE = eval(model, train_df)
    valid_MAE_normalized, valid_MAE, valid_MSE_normalized, valid_MSE, = eval(model, valid_df)
    print("EPOCH:\t" + str(epoch) + '\n' \
          #         +"train_MAE_normalized: "+str(train_MAE_normalized)+'\n'\
          #         +"valid_MAE_normalized: "+str(valid_MAE_normalized)+'\n'\
          + "train_MAE" + ":" + "\n" + str(train_MAE) + '\n' \
          + "valid_MAE" + ":" + "\n" + str(valid_MAE) + '\n' \
\
          + "train_MSE_normalized_mean: " + str(train_MSE_normalized.mean()) + '\n' \
          + "valid_MSE_normalized_mean: " + str(valid_MSE_normalized.mean()) + '\n' \
          #         +"train_MSE_normalized: "+str(train_MSE_normalized)+'\n'\
          #         +"valid_MSE_normalized: "+str(valid_MSE_normalized)+'\n'\
          )
    if train_MSE_normalized.mean() < best_param["train_MSE_normalized"]:
        best_param["train_epoch"] = epoch
        best_param["train_MSE_normalized"] = train_MSE_normalized.mean()
    if valid_MSE_normalized.mean() < best_param["valid_MSE_normalized"]:
        best_param["valid_epoch"] = epoch
        best_param["valid_MSE_normalized"] = valid_MSE_normalized.mean()
        if valid_MSE_normalized.mean() < 0.4:
            test_MAE_normalized, test_MAE, test_MSE_normalized, test_MSE, test_r2_l, test_mae, test_rmse, test_r2 = eval(
                model, test_df)
            torch.save(model, sys.argv[
                1] + 'saved_models/model' + '_' + str(epoch) + '.pt')

    if (epoch - best_param["train_epoch"] > 10) and (epoch - best_param["valid_epoch"] > 18):
        break

print('ENDE')