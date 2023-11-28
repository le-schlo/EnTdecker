import os
import json
import time
import torch

import numpy as np
import pandas as pd

from EasyChemML.DataImport.DataImporter import DataImporter
from EasyChemML.DataImport.Module.CSV import CSV
from EasyChemML.Encoder import BertTokenizer, MolRdkitConverter
from EasyChemML.Environment import Environment
from EasyChemML.Model.impl_Pytorch.Models.BERT.FP2MOL_BERT_Trans import FP2MOL_Bert

# ----------------------------------- Data Preprocessing -----------------------
os.environ["HDF5_USE_FILE_LOCKING"] = 'False'
settings_path ="settings.json"

with open(settings_path, "r") as settings_file:
    setting_dict = json.load(settings_file)

device = "cuda:0"

d_model = 512
heads = 4
N = 8
src_vocab_size = setting_dict.get("src_vocab_size")
trg_vocab_size = 136
dropout = 0.1
max_seq_len = setting_dict.get("src_len")

dir_name = setting_dict.get("dir_name")
file_path = setting_dict.get("file_path")

model_object = FP2MOL_Bert(src_vocab_size, trg_vocab_size, N, heads, d_model, dropout, max_seq_len, device)
model_object.init_internal_parameters()

env = Environment(WORKING_path_addRelativ='Output')

smiSD_loader = {'EnTdecker_data': CSV(file_path, columns=['SMILES', 'SMILES_SD'])}
di = DataImporter(env)
bp = di.load_data_InNewBatchPartition(smiSD_loader)

train_data_size = len(bp['EnTdecker_data'])



print('Start BertTokenizer')
tokenizer = BertTokenizer()
tokenizer.convert(datatable=bp['EnTdecker_data'], columns=['SMILES', 'SMILES_SD'], n_jobs=4)
# ----------------------------------- Training -----------------------
EPOCHS = setting_dict.get("epochs")
batch_size = setting_dict.get("batch_size")
print_every = setting_dict.get("print_every")
training_steps = 0
loss_list = []
training_steps_list = []

for count in range(EPOCHS):

    start = time.time()
    temp = start

    total_loss = 0
    epoch = count + 1
    iteration_per_chunk = train_data_size / batch_size

    start_mol = 0

    for i in range(int(iteration_per_chunk)):
        training_time = time.time()
        end_mol = start_mol + batch_size

        smiles_sd_ids = bp['EnTdecker_data'][start_mol:end_mol]['SMILES_SD_ids']
        smiles_ids = bp['EnTdecker_data'][start_mol:end_mol]['SMILES_ids']
        smiles_ids = np.array([smiles_ids[i][0:setting_dict.get("src_len")] for i in range((batch_size))])
        smiles_ids = smiles_ids[:][0:setting_dict.get("src_len")]
        print('train_time = ', time.time() - training_time)

        loss = model_object.fit_CalcLoss(smiles_ids, smiles_sd_ids)
        total_loss += torch.Tensor.item(loss.data)
        start_mol = end_mol
        training_steps += 1
        loss_avg = total_loss / (i + 1)

        if (i + 1) % print_every == 0:
            print(f"time = {(time.time() - start) // 60}, epoch {epoch}, iter = {i + 1}, loss = {loss_avg: .5f}, "
                  f"{(time.time() - temp): .3f}s per {print_every}")
            temp = time.time()

        if training_steps % 100 == 0:
            loss_list.append(loss_avg)
            training_steps_list.append(training_steps)
            loss_store_df = pd.DataFrame({'training_steps': training_steps_list, 'training_loss': loss_list})
            loss_store_df.to_csv(dir_name + (setting_dict.get("loss_fname")))

p_fname = dir_name + ("model_p_checkpoint.pt")
o_fname = dir_name + ("model_o_checkpoint.pt")
model_object.save_model(p_fname, o_fname)
print('ENDE')
