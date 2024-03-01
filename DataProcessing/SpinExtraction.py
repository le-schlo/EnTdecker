import numpy as np
import re

def returnSmileswithSpinDensity(smi, spin_population):
    '''
    Given a smiles and an array of spin populations, this function returns a smiles with the spin density encoded in a string.

    :param smi: SMILES
    :param spin_population: array of spin populations, containing only the spin populations of the heavy atoms
    :return: a string with the spin density encoded in the SMILES
    '''

    invalid_characters=['=', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                        '(', ')', '[', ']', '@', 'H', '#', '/', "\\", '-', '+', '[H]']
    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|_0_|_1_|_2_|_3_|_4_|_5_|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

    bins = np.array([-1,0.07,0.2,0.4,0.8,2])
    digitized = np.digitize(spin_population, bins)

    split_by_tokens = re.compile(SMI_REGEX_PATTERN)
    tokens = [token for token in re.findall(split_by_tokens,smi)]

    separator1='_'
    separator2='_'
    strwithSD=''
    counter=0

    for character in tokens:
        strwithSD +=character
        if character not in invalid_characters:
            strwithSD += separator1+str(digitized[counter])+separator2
            counter+=1

    return strwithSD
