import pandas as pd
import numpy as np
import pickle
drug_comb = pd.read_csv("data_raw/summary_v_1_5.csv", usecols=["drug_row", "drug_col", "cell_line_name", "tissue_name"], dtype="category")
print(drug_comb.shape, drug_comb.memory_usage().sum()/1e6)
drug_comb = drug_comb[drug_comb.cell_line_name=="MDA-MB-231"]
print(drug_comb.shape)
comb = pd.unique(pd.concat([drug_comb.drug_row, drug_comb.drug_col]).dropna()).astype("str")
comb = pd.Series(comb, dtype="str")

mask_semicolon = comb.str.contains("\S+:\S+(?=\s?)")

splitted_semicolon = comb.str.extract("(\S+:\S+(?=\s?))").dropna().squeeze().str.split(":")

comb = pd.concat([
    comb[~comb.str.contains("\S+:\S+(?=\s?)")],
    splitted_semicolon.str.get(0),
    splitted_semicolon.str.get(1)
]).drop_duplicates()
comb.to_csv("data/drugcomb_drugs.csv", index=False)
