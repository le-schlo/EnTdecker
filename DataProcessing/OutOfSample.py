import pandas as pd
from rdkit import Chem


df = pd.read_csv('/Data/EnTdecker_data.csv')

match = []
for smi in df.smiles.to_list():
    patt = Chem.MolFromSmarts('c1ccc(cc1)C=C')

    if Chem.MolFromSmiles(smi).HasSubstructMatch(patt):
        match.append(True)
    else:
        match.append(False)

df['OoS']=match
df_test = df[df['OoS']==True]
df_train = df[df['OoS']==False]


