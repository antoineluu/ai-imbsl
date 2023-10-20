import pandas as pd
import re
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import cm


def replace_cell_names_with_id(dataframe: pd.DataFrame, mapping_file:str ="../data/mappingccl.csv", cell_col:str ="cell_line"):
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

def plot_from_df(data, model, device, drug_data, cell_data, title="Prediction over full experimental design (88 epochs)", test=False, ssize =30):
    cell_mapping = pd.read_csv("../data/mappingccl.csv", usecols=["CCLE_Name","Broad_ID"]).drop_duplicates("Broad_ID")
    cell_mapping = cell_mapping.set_index(keys="Broad_ID")
    
    # ssize = 30
    fig = plt.figure(figsize=(15,15))
    fig.tight_layout()

    ax = fig.add_subplot(projection='3d')


    ax.stem(data.drugA_conc, data.drugB_conc, data.target, markerfmt='or', linefmt="--b")
    Z = np.empty((ssize,ssize))
    A = np.linspace(data.drugA_conc.min(),data.drugA_conc.max(), ssize)
    B = np.linspace(data.drugB_conc.min(),data.drugB_conc.max(), ssize)
    if test:
        A = np.linspace(0.01,data.drugA_conc.max(), ssize)
        B = np.linspace(0.01,data.drugB_conc.max(), ssize)
    X,Y = np.meshgrid(A,B)
    # Conc = np.concatenate([X,Y]).reshape((-1,2), order='C')
    for m in range(X.shape[0]):
        for n in range(X.shape[1]):
            sample_comb = data.iloc[0:1, 0:3]
            sample_comb["drugA_conc"] = X[m,n]
            sample_comb["drugB_conc"] = Y[m,n]
            sample_comb["target"] = data.iloc[0:1, 5:6]
            sample_set  = Dataset_from_pd(sample_comb, drug_data, cell_data)
            sample, _ = next(iter(DataLoader(sample_set)))
            sample = sample.to(device)
            predict = model(sample)
            Z[m,n] = predict

    # ax.scatter()
    # print(cell_mapping.loc[data.cell_line.iloc[0]])
    # Plot a basic wireframe.
    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))
    rcount, ccount, _ = colors.shape
    surf = ax.plot_surface(X, Y, Z, rcount=rcount, ccount=ccount, facecolors=colors, shade=False)
    surf.set_facecolor((0,0,0,0))
    ax.set_xlabel("{} (muM)".format(data.drugA_name.iloc[0]), size=25)
    ax.set_ylabel("{} (muM)".format(data.drugB_name.iloc[0]), size=25)
    ax.set_zlabel("Cell viability X/X0", size=23)
    ax.set_title("Cell line : " +str(cell_mapping.loc[data.cell_line.iloc[0]].to_numpy()[0]), size=25)
    ax.view_init(25, 35)
    # plt.suptitle(title, size=25)

def pcc_fn(outputs, labels):

        vx = outputs - torch.mean(outputs)
        vy = labels - torch.mean(labels)
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
def train_one_epoch(model, epoch_index, tb_writer, training_loader, optimizer, loss_fn, device, L1=0, verbose=False, print_every=10):
    running_loss = 0.
    last_loss = 0.
    model = model.to(device)
    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.to(device=device)
        labels = labels.to(device=device)
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        outputs = model.forward(inputs)

        params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_regularization = L1 * torch.linalg.vector_norm(params, 1)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels) + l1_regularization
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        running_loss += loss.item()
        # print(running_loss)
        if i % print_every == print_every-1:
            last_loss = running_loss / print_every # loss per batch
            # print('  batch {} loss: {}'.format(i + 1, last_loss), outputs[0][0].item(), labels[0][0].item())
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
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
    def __init__(self, h_sizes, drug_length, cell_length):
        super().__init__()
        self.drug_length = drug_length
        self.cell_length = cell_length
        self.drug_encoder = Encoder(h_sizes=[drug_length, 512, 512, 256, 512, 512])
        self.cell_encoder = Encoder(h_sizes=[cell_length, 512, 512, 256, 512, 512])
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.hidden.append(nn.Dropout(0.1))
            self.hidden.append(nn.ReLU())
        self.hidden.append(nn.Linear(h_sizes[-1], 1))

    def forward(self, x):
        drug_A, drug_B, cell, drugA_conc, drugB_conc = torch.split(x, [self.drug_length, self.drug_length, self.cell_length, 1, 1], dim=1)
        drug_A_emb = self.drug_encoder(drug_A)
        drug_B_emb = self.drug_encoder(drug_B)
        cell_emb = self.cell_encoder(cell)
        x = torch.concatenate([drug_A_emb, drug_B_emb, cell_emb, drugA_conc, drugB_conc], dim=1)
        for lay in self.hidden:
            # print(torch.sum(torch.isnan(x)))
            # print(torch.sum(x>1e3))
            x = lay(x)
        return x
    
def prepare_train_val(df_train, unique_pairs):
    train_unique_pairs, val_unique_pairs = train_test_split(unique_pairs, test_size=0.2, random_state=42)
    combined_train = train_unique_pairs.loc[:,"drugA_name"].str.cat(train_unique_pairs.loc[:,"drugB_name"], sep= " + ")
    combined_val = val_unique_pairs.loc[:,"drugA_name"].str.cat(val_unique_pairs.loc[:,"drugB_name"], sep= " + ")

    data_train = df_train[df_train.loc[:,"drugA_name"].str.cat(df_train.loc[:,"drugB_name"],sep=" + ").isin(combined_train)]
    data_val = df_train[df_train.loc[:,"drugA_name"].str.cat(df_train.loc[:,"drugB_name"],sep=" + ").isin(combined_val)]

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