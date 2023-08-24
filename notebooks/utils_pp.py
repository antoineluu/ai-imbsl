import pandas as pd
import re

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