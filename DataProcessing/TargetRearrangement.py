import re
import random
import numpy as np
from rdkit import Chem
#--------Functions to rearrange the string with SMILES and the respective triplet energies-------------#
def getRandomisedSmilesandSD(initial_smiles, initial_spin_population, num=5):
    # Main Function
    """
    Function to rearrange the string with SMILES and the respective spin populations
    :param initial_smiles: SMILES to be rearranged
    :param initial_spin_population: corresponding spin population
    :param num: number of rearrangements to be conducted
    :return: two lists: I) rearranged SMILES and II) rearranged SMILES with incorporated spin population
    """
    splited_notokens = getListofTokens(initial_spin_population)

    DensityArray = getDensityArray(splited_notokens)

    new = []
    for i in range(num):
        rearranged_smiles, rearranged_SD = getRearrangedSmilesandSD(initial_smiles, DensityArray)

        rearranged_target_string = return_smiSD(rearranged_smiles, rearranged_SD)
        if rearranged_target_string not in new:
            new.append(rearranged_target_string)

    new_smi = []
    new_densitystring = []
    for smiSD in new:
        smistring = ''
        densitystring = ''
        DensityArray_n = []
        for tok in getListofTokens(smiSD):
            if '_' not in tok:
                smistring += tok
            if '_' in tok:
                DensityArray_n.append(int(tok.strip('_')))
            densitystring += tok

        new_smi.append(smistring)
        new_densitystring.append(densitystring)

    return new_smi, new_densitystring

#--------Helper functions for getRandomisedSmilesandSD()-------------#
def getListofTokens(smi: str)->list:
    """
    Function to split the string with SMILES into a list of tokens
    :param smi: SMILES string
    :return: list of tokens
    """
    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|(?:_\d{..})|(?:_\d{.})|_0_|_1_|_2_|_3_|_4_|_5_|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    reg_splitter = re.compile(SMI_REGEX_PATTERN)
    tokens = [token for token in reg_splitter.findall(smi)]
    return tokens

def getDensityArray(tokens):
    """
    Function to extract the array of spin population from the list of tokens
    :param splited_notokens: list of tokens
    :return: array of spin population
    """
    DensityArray = []
    for tok in tokens:
        if '_' in tok:
            DensityArray.append(int(tok.strip('_')))
    return np.array(DensityArray)

def getRearrangedSmilesandSD(initial_smiles, density_array):
    """
    Function to rearrange the string with SMILES and the respective spin population
    :param initial_smiles: Initial SMILES
    :param density_array: Spin population array
    :return: Rearranged SMILES and rearranged spin population array
    """
    initial_mol = Chem.MolFromSmiles(initial_smiles)
    master_dict={}
    for idx, atom in enumerate(initial_mol.GetAtoms()):
        atom.SetAtomMapNum(idx+1)
        master_dict[idx+1]=density_array[idx]


    smi_w_MapNum = randomSmiles(initial_mol)
    new_mol = Chem.MolFromSmiles(smi_w_MapNum)

    MapNum=[atom.GetAtomMapNum() for atom in new_mol.GetAtoms()]

    new_SDArray = []
    for mn in MapNum:
        new_SDArray.append(master_dict[mn])

    [atom.SetAtomMapNum(0) for atom in new_mol.GetAtoms()]
    smi_wo_MapNum = Chem.MolToSmiles(new_mol,canonical=False)

    return smi_wo_MapNum, new_SDArray

def randomSmiles(mol):
    '''
    Function to output a rearranged SMILES string of the input molecule
    :param mol:
    :return:
    '''

    mol.SetProp("_canonicalRankingNumbers", "True")
    idxs = list(range(0, mol.GetNumAtoms()))
    random.shuffle(idxs)
    for i,idx in enumerate(idxs):
        mol.GetAtomWithIdx(i).SetProp("_canonicalRankingNumber", str(idx))
    return Chem.MolToSmiles(mol)

def return_smiSD(smiles, spin_population):
    """
    Similar to returnSmileswithSpinDensity() from DataProcessing\SpinExtraction.py but without binning the spin population
    :param smiles: SMILES string
    :param spin_population: array of spin population
    :return: a string with the spin density encoded in the SMILES
    """
    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|(?:_\d{..})|(?:_\d{.})|_0_|_1_|_2_|_3_|_4_|_5_|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    invalid_characters = ['=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                          '(', ')', '[', ']', '@', 'H', '#', '/', "\\", '-', '+']

    spl = re.compile(SMI_REGEX_PATTERN)

    tokens = [token for token in re.findall(spl, smiles)]

    separator1 = '_'
    separator2 = '_'
    target_string = ''
    counter = 0
    for character in tokens:

        target_string += character
        if character not in invalid_characters:
            target_string += separator1 + str(spin_population[counter]) + separator2
            counter += 1

    return target_string