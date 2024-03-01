import sys
import os
import json
import time
import torch

from pathlib import Path

import numpy as np
import pandas as pd

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.CSV import CSV
from EasyChemML.Encoder import BertTokenizer, MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.Encoder.impl_Tokenizer.SmilesTokenizer_SchwallerEtAll import SmilesTokenzier
from EasyChemML.Metrik.Module.DensityEvaluation import DensityMetrics
from EasyChemML.Model.impl_Pytorch.Models.BERT.FP2MOL_BERT_Trans import FP2MOL_Bert

# ----------------------------------- Data Preprocessing -----------------------
settings_path ="/Models/spin_population/settings.json"
with open(settings_path, "r") as settings_file:
    setting_dict = json.load(settings_file)

dir_name = setting_dict.get("dir_name")
file_path = setting_dict.get("file_path")

device = "cuda:0"

d_model = 512
heads = 4
N = 8
src_vocab_size = setting_dict.get("src_vocab_size")
trg_vocab_size = 111
dropout = 0.1
max_seq_len = setting_dict.get("src_len")

model_object = FP2MOL_Bert(src_vocab_size, trg_vocab_size, N, heads, d_model, dropout, max_seq_len, device)
p_fname = "/Models/spin_population/spin_population.pt"
model_object.load_model_parameters(p_fname)

env = Environment(WORKING_path_addRelativ='Output')



dataloader = {'EnTdecker_data': CSV(file_path, columns=['SMILES', 'SMILES_SD'])}
di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(dataloader)

val_data_size = len(bp['EnTdecker_data'])

print('Start BertTokenizer')
tokenizer = BertTokenizer()
tokenizer.convert(datatable=bp['EnTdecker_data'], columns=['SMILES', 'SMILES_SD'], n_jobs=4)

# ----------------------------------- Evaluation -----------------------

batch_size = setting_dict.get("batch_size")
print_every = setting_dict.get("print_every")

s = SmilesTokenzier()

total_loss = 0
positive_prediction = []
smi2smi = []
smi2smi_l = []
R2_perMol = []
R2_perMol_l = []
Ranked = []
Ranked_l = []

start_mol = 0
iter_steps = 0

start = time.time()
temp = start

iteration_per_chunk = val_data_size / batch_size

for i in range(int(iteration_per_chunk)):
    end_mol = start_mol + batch_size

    true_smiSD = bp['EnTdecker_data'][start_mol:end_mol]['SMILES_SD_ids']
    input_smi = bp['EnTdecker_data'][start_mol:end_mol]['SMILES_ids']
    input_smi = input_smi[:, 0:setting_dict.get("src_len")]

    true_smi = bp['EnTdecker_data'][start_mol:end_mol]['SMILES']

    true_smiSD = torch.LongTensor(true_smiSD).to(torch.device(device))
    input_smi = torch.LongTensor(input_smi).to(torch.device(device))
    model_eval = model_object.fit_eval(input_smi, true_smiSD, method='greedy')
    loss, outputs = model_eval.loss, model_eval.outputs
    total_loss += torch.Tensor.item(loss.data)
    start_mol = end_mol
    iter_steps += 1

    for example in range(batch_size):

        pred_smi, pred_SD_string, DensityArray = s.getSmilesfromoutputwithSD(outputs[example])
        _, _, true_DensityArray = s.getSmilesfromoutputwithSD(true_smiSD[example])

        if torch.equal(outputs[example], true_smiSD[example]):
            positive_prediction.append(1)
        else:
            positive_prediction.append((0))

        if pred_smi == true_smi[example].decode("utf-8"):
            smi2smi.append(1)
        else:
            smi2smi.append(0)

        num_highest = 10
        Arrays4metric = DensityMetrics(true_DensityArray, DensityArray)
        R2_perMol.append(Arrays4metric.PearsonR2_np())
        Ranked.append(Arrays4metric.RankDensities(num_highest=num_highest))

    if (i + 1) % print_every == 0:
        accuracy = np.sum(positive_prediction) / (iter_steps * batch_size)
        smi_acc = np.sum(smi2smi) / (iter_steps * batch_size)
        R2_acc = np.sum(R2_perMol) / (iter_steps * batch_size)
        Ran = np.sum(Ranked) / (iter_steps * batch_size)

        print(f'time = {(time.time() - start) // 60}, '
              f'exact accuracy = {accuracy: .3f}, ' f'smi2smi accuracy = {smi_acc: .3f}, '
              f'Top{num_highest/2} ' f'Ranked Score = {Ran: .3f}, '
              f'Average R2 = {R2_acc: .3f}, ' f'{(time.time() - temp): .3f}s per {print_every}')

        temp = time.time()


positive_prediction.append(accuracy)
smi2smi_l.append(smi_acc)
R2_perMol_l.append(R2_acc)
Ranked_l.append(Ran)


loss_store_df = pd.DataFrame({'positive_prediction': positive_prediction,
                              'smi2smi': smi2smi_l,
                              'R2 per mol': R2_perMol_l,
                              'Ranked': Ranked_l,
                              })

loss_store_df.to_csv(dir_name + 'eval.csv')

print('well_done')
