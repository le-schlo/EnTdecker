import random

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdmolops import FastFindRings

from collections import defaultdict

def StructuralSplitter_Murcko(dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=42):
    '''
    This code for splitting based on the Murcko scaffolds is based on the implementation of Li et al. (https://doi.org/10.1021/acsomega.1c04017)
    and can be found in their public github repo (https://doi.org/10.1021/acsomega.1c04017).

    :param dataset: Dataset to be split according to the molecules scaffolds. Should be a pandas dataframe with the molecules as SMILES in a column named 'smiles'
    :param frac_train: fraction of the dataset to be used for training
    :param frac_val: fraction of the dataset to be used for validation
    :param frac_test: fraction of the dataset to be used for testing
    :param random_state: random seed for shuffling the order of the scaffolds

    :return: Lists of indices of the molecules in the training, validation and test sets
    '''
    smis = dataset.smiles.tolist()
    molecules = [Chem.MolFromSmiles(x) for x in smis]

    scaffolds = defaultdict(list)
    scaffold_func = 'decompose'
    dictidxsmi = {}
    for i, mol in enumerate(molecules):
        dictidxsmi[i] = Chem.MolToSmiles(mol)
        # For mols that have not been sanitized, we need to compute their ring information
        try:
            FastFindRings(mol)
            if scaffold_func == 'decompose':
                mol_scaffold = Chem.MolToSmiles(AllChem.MurckoDecompose(mol))
            if scaffold_func == 'smiles':
                mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            # Group molecules that have the same scaffold
            smi = Chem.MolToSmiles(mol)
            scaffolds[mol_scaffold].append(i)
        except:
            print('Failed to compute the scaffold for molecule {:d} '
                  'and it will be excluded.'.format(i + 1))

    # Order groups of molecules by first comparing the size of groups
    # and then the index of the first compound in the group.
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    random.seed(random_state)
    random.shuffle(scaffold_sets)

    train_indices, val_indices, test_indices = [], [], []
    train_cutoff = int(frac_train * len(molecules))
    val_cutoff = int((frac_train + frac_val) * len(molecules))

    for group_indices in scaffold_sets:

        if len(train_indices) + len(group_indices) > train_cutoff:
            if len(train_indices) + len(val_indices) + len(group_indices) > val_cutoff:
                test_indices.extend(group_indices)
            else:
                val_indices.extend(group_indices)
        else:
            train_indices.extend(group_indices)

    train_set = []
    for idx in train_indices:
        train_set.append(dictidxsmi[idx])
    validation_set = []
    for idx in val_indices:
        validation_set.append(dictidxsmi[idx])
    test_set = []
    for idx in test_indices:
        test_set.append(dictidxsmi[idx])
    return sorted(list(train_indices)), sorted(list(val_indices)), sorted(list(test_indices))