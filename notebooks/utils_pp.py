import pandas as pd
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def replace_cell_names_with_id(dataframe: pd.DataFrame, mapping_file:str ="data/mappingccl.csv", cell_col:str ="cell_line"):
    cell_mapping = pd.read_csv(mapping_file, usecols=["Aliases", "CCLE_Name", "Broad_ID"])
    clean = lambda s: str(s).replace('-','').replace(' ','').replace('_','').replace(':','').replace(';','').upper()
    alias2id={}
    for cell_name_raw in pd.unique(dataframe.loc[:,cell_col]):
        count=0
        cell_name = clean(cell_name_raw)
        for i in cell_mapping.index:
            alias, CCLE_Name = map(clean,[cell_mapping.loc[i, "Aliases"], cell_mapping.loc[i, "CCLE_Name"]] )
            if re.search(cell_name,alias) or re.search(cell_name,CCLE_Name) or re.search(alias, cell_name) or re.search(CCLE_Name, cell_name):
                matching_id=cell_mapping.loc[i,"Broad_ID"]
                count+=1
        alias2id[cell_name_raw]=matching_id if count==1 else None
    dataframe_1 = dataframe.copy(True)
    dataframe_1.loc[:,cell_col] = dataframe.loc[:,cell_col].map(alias2id)
    dataframe_1.dropna(inplace=True)
    return dataframe_1
    
# def merge_input_data(df, drugA="drugA_name", drugB="drugB_name", cell_line="cell_line"):
#     X = pd.merge(left=df, right=smile.rename(columns={'name':'drugA_name'} |{column:column+"_A" for column in smile.columns[1:]}),how="inner",on="drugA_name")
#     X = pd.merge(left=X, right=smile.rename(columns={'name':'drugB_name'} |{column:column+"_B" for column in smile.columns[1:]}),how="inner",on="drugB_name")
#     X.insert(1,"cell_line_id",X.loc[:,"cell_line"].apply(lambda x:alias2id[x]))
#     X = X.merge(right=cell_data, how="inner", left_on="cell_line_id", right_index=True)
#     y = X["target"]
#     X.drop(columns="target")
#     return X.reset_index(drop=True), y.reset_index(drop=True)

class Encoder(torch.nn.Module):
    def __init__(self, h_sizes=None, dropout=0.2):
        self.h_sizes = h_sizes
        super().__init__()
        if h_sizes is None: h_sizes = [32,32,32,32]
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(h_sizes[0], h_sizes[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h_sizes[1], h_sizes[2]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(h_sizes[2], h_sizes[3])
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class AE_DNN(torch.nn.Module):
    def __init__(self, h_sizes):
        super().__init__()
        self.drug_encoder = Encoder(h_sizes=[drug_length, 512, 512, 256, 512, 512])
        self.cell_encoder = Encoder(h_sizes=[cell_length, 512, 512, 256, 512, 512])
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(nn.Dropout(0.1))
            self.hidden.append(nn.ReLU())
        self.hidden.append(nn.Linear(h_sizes[-1], 1))
    def forward(self, x):
        drug_A, drug_B, cell, drugA_conc, drugB_conc = torch.split(x, [drug_length, drug_length, cell_length, 1, 1], dim=1)
        drug_A_emb = self.drug_encoder(drug_A)
        drug_B_emb = self.drug_encoder(drug_B)
        cell_emb = self.cell_encoder(cell)
        x = torch.concatenate([drug_A_emb, drug_B_emb, cell_emb, drugA_conc, drugB_conc], dim=1)
        for lay in self.hidden:
            # print(torch.sum(torch.isnan(x)))
            # print(torch.sum(x>1e3))
            x = lay(x)
        return x

class Dataset_from_pd(Dataset):
    def __init__(self, drug_comb_data, drug_feat, cell_feat):
        self.drug_comb_data = drug_comb_data.to_numpy()
        self.drug_feat = drug_feat.to_numpy()
        self.cell_feat = cell_feat.to_numpy()
        self.drug_mapping = pd.Series(range(len(self.drug_feat)), index=drug_feat.index).to_dict()
        self.cell_mapping = pd.Series(range(len(self.cell_feat)), index=cell_feat.index).to_dict()
        # print(self.cell_mapping, self.drug_mapping)

        print()
    def __len__(self):
        return len(self.drug_comb_data)
    
    def __getitem__(self, idx):
        combi = self.drug_comb_data[idx]
        drug_A = self.drug_feat[self.drug_mapping[combi[1]]]
        drug_B = self.drug_feat[self.drug_mapping[combi[2]]]
        cell_line = self.cell_feat[self.cell_mapping[combi[0]]]


        return np.concatenate([drug_A, drug_B, cell_line, combi[3:5].astype("float32")], dtype="float32"), combi[5:6].astype("float32")

train_set  = Dataset_from_pd(df_train, drug_data, cell_data)
train_dl = DataLoader(train_set, batch_size=256, shuffle=True)
xi, yi = next(iter(train_dl))
print(xi.shape, yi.shape)
# print(np.argwhere(xi>1e2))
# print(tuple(np.argwhere(xi>100)))
print(xi.numpy()[tuple(np.argwhere(xi>100))])
# print(xi.numpy()[:,tuple(np.argwhere(xi>100))[1]])

if __name__=="__main__":

    columns = ["cell_line", "drugA_name", "drugB_name", "drugA_conc", "drugB_conc", "target"]
    df_train = pd.read_csv("data/oneil.csv", usecols=(1,2,3,4,5,12)).iloc[:,[0,1,3,2,4,5]]
    df_test = pd.read_csv("data/test_yosua.csv").iloc[:,[2,3,0,1,4,5]]
    df_test.insert(0,"cell_line",pd.Series(["MDA_MB_231" for _ in range(df_test.shape[0])]))
    df_test = df_test.set_axis(columns+["uncertainty"], axis=1).apply(lambda series:series.astype("Float32") if series.dtype=="float64" else series)
    df_train = df_train.set_axis(columns, axis=1).apply(lambda series:series.astype("Float32") if series.dtype=="float64" else series)
    smile = pd.read_csv("data/smile.csv").apply(lambda series:series.astype("Float32") if series.dtype=="float64" else series)
    cell_data = pd.read_csv("data/cell_line_data.csv", index_col=0).astype("Float32")
    print("all files loaded")