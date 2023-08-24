import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import re
import numpy as np
types=defaultdict(lambda : "Float32")
types[0]="string"
scaler = StandardScaler()

data = pd.read_pickle("data/cell_lines_omics.pkl.compress", compression="gzip")
data.set_index(keys='Unnamed: 0', drop=True, inplace=True)
std = data.std(axis=0)
qt = data.quantile(0.75,interpolation='nearest')
data = data.loc[:,qt>1.2]
data = data.loc[:,std>1]
data_np = scaler.fit_transform(data)
export = pd.DataFrame(data_np, index=data.index.astype("category"))
print(export.isna().sum().sum())
export = export.apply(pd.to_numeric, downcast="float")
export.to_pickle("data/cell_line_data.pkl.compress", compression="gzip")