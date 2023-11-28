from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import numpy as np
import random

def SimilaritySplit(
    dataset,
    frac_train: float = 0.8,
    frac_valid: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42):
    """
        This code for splitting the dataset is taken from the DeepChem library. Their code is publicly avaialble on GitHub (https://github.com/deepchem/deepchem/tree/master#citing-deepchem)

        Splits compounds into training, validation, and test sets based on the
        Tanimoto similarity of their ECFP4 fingerprints. This splitting algorithm
        has an O(N^2) run time, where N is the number of elements in the dataset.
        Parameters
        ----------
        dataset: Dataset
            Dataset to be split. Should be a pandas DataFrame with a molecules stored as SMILES in a column named "smiles"
        frac_train: float, optional (default 0.8)
            The fraction of data to be used for the training split.
        frac_valid: float, optional (default 0.1)
            The fraction of data to be used for the validation split.
        frac_test: float, optional (default 0.1)
            The fraction of data to be used for the test split.
        seed: int, optional (default None)
            Random seed to use (ignored since this algorithm is deterministic).
        log_every_n: int, optional (default None)
            Log every n examples (not currently used).
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            A tuple of train indices, valid indices, and test indices.
        """

    # Compute fingerprints for all molecules.
    mols=[]
    fps=[]
    for i, smiles in enumerate(dataset.smiles.tolist()):
        m = Chem.MolFromSmiles(smiles)
        mols.append(m)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024))

        print(str(i) + ' / ' + str(len(dataset)))

    # Split into two groups: training set and everything else.

    train_size = int(frac_train * len(dataset))
    valid_size = int(frac_valid * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_inds, test_valid_inds = _split_fingerprints(
        fps, train_size, valid_size + test_size)

    # Split the second group into validation and test sets.

    if valid_size == 0 or frac_valid == 0:
        valid_inds = []
        test_inds = test_valid_inds
    elif test_size == 0 or frac_test == 0:
        test_inds = []
        valid_inds = test_valid_inds
    else:
        test_valid_fps = [fps[i] for i in test_valid_inds]
        test_inds, valid_inds = _split_fingerprints(test_valid_fps,
                                                    test_size, valid_size)
        test_inds = [test_valid_inds[i] for i in test_inds]
        valid_inds = [test_valid_inds[i] for i in valid_inds]
    return sorted(list(train_inds)), sorted(list(valid_inds)), sorted(
        list(test_inds))

def _split_fingerprints(fps: list, size1: int,
                        size2: int):
    """This is called by FingerprintSplitter to divide a list of fingerprints into
    two groups.
    """
    assert len(fps) == size1 + size2


    # Begin by assigning the first molecule to the first group.

    fp_in_group = [[fps[0]], []]
    indices_in_group: tuple[list[int], list[int]] = ([0], [])
    remaining_fp = fps[1:]
    remaining_indices = list(range(1, len(fps)))
    max_similarity_to_group = [
        DataStructs.BulkTanimotoSimilarity(fps[0], remaining_fp),
        [0] * len(remaining_fp)
    ]
    # Return identity if no tuple to split to
    if size2 == 0:
        return ((list(range(len(fps)))), [])

    while len(remaining_fp) > 0:
        # Decide which group to assign a molecule to.

        group = 0 if len(fp_in_group[0]) / size1 <= len(
            fp_in_group[1]) / size2 else 1

        # Identify the unassigned molecule that is least similar to everything in
        # the other group.

        i = np.argmin(max_similarity_to_group[1 - group])

        # Add it to the group.

        fp = remaining_fp[i]
        fp_in_group[group].append(fp)
        indices_in_group[group].append(remaining_indices[i])

        # Update the data on unassigned molecules.

        similarity = DataStructs.BulkTanimotoSimilarity(fp, remaining_fp)
        max_similarity_to_group[group] = np.delete(
            np.maximum(similarity, max_similarity_to_group[group]), i)
        max_similarity_to_group[1 - group] = np.delete(
            max_similarity_to_group[1 - group], i)
        del remaining_fp[i]
        del remaining_indices[i]
    return indices_in_group