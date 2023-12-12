import pandas as pd
import numpy as np

all = pd.read_csv("/Data/EnT_data.csv")

all = all.sample(frac=1)

chunks = np.array_split(all, 5)

for i, seed in enumerate([42,101,143,404,666]):
    train = pd.concat([chunks[j] for j in range(5) if j != i])
    testvalid = chunks[i]
    valid = testvalid.sample(frac=0.5, random_state=seed)
    test = testvalid.drop(valid.index)
                       
    train.to_csv('%s_train.csv' % seed, index=False)
    valid.to_csv('%s_valid.csv' % seed, index=False)
    test.to_csv('%s_test.csv' % seed, index=False)
