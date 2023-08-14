import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import re
import numpy as np
types=defaultdict(lambda : "Float32")
types[0]="string"
scaler = StandardScaler()

# cell_mapping = pd.read_csv("data/mappingccl.csv")
# cell_names = np.concatenate(((pd.read_csv("data/oneil.csv", usecols=[1])).squeeze().unique(),["MDA_MB_231"]))
# clean = lambda s: str(s).replace('-','').replace(' ','').replace('_','').replace(':','').replace(';','').upper()
# id2alias = dict()
# for cell_name in cell_names:
#     count=0
#     cell_name_clean = clean(cell_name)
#     for i in cell_mapping.index:
#         alias = clean(cell_mapping.loc[i, "Aliases"])
#         CCLE_Name = clean(cell_mapping.loc[i, "CCLE_Name"])
#         # print(alias, cell_name_clean, CCLE_Name)
#         if re.search(cell_name_clean,alias) or re.search(cell_name_clean,CCLE_Name) or re.search(alias,cell_name_clean) or re.search(CCLE_Name,cell_name_clean):
#             print(cell_name, alias)
#             matching_id=cell_mapping.loc[i,"Broad_ID"]
#             count+=1
#     id2alias[matching_id]=cell_name if count==1 else None

data = pd.read_pickle("data/cell_lines_omics.pkl.compress", compression="gzip")
data.set_index(keys='Unnamed: 0', drop=True, inplace=True)
std = data.std(axis=0)
qt = data.quantile(0.75,interpolation='nearest')
data = data.loc[:,qt>1.2]
data = data.loc[:,std>1]
data_np = scaler.fit_transform(data)

pd.DataFrame(data_np, index=data.index).to_pickle("data/cell_line_data.pkl.compress", compression="gzip")
