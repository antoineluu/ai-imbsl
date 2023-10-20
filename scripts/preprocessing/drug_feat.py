
import pandas as pd
import numpy as np
import requests
import re
import pickle
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors


fpgen = AllChem.GetMorganGenerator(radius=3)
oneil = pd.read_csv("data_raw/oneil.csv", usecols=["drugA_name", "drugB_name"], dtype="str")
yosua = pd.read_csv("data/test_yosua.csv", usecols=["drugA_name", "drugB_name"], dtype = "str")
drugcomb_drugs = pd.read_csv("data/drugcomb_drugs.csv", dtype="str").squeeze()
identifiers = pd.concat([oneil.iloc[:,0],oneil.iloc[:,1],yosua.iloc[:,0],yosua.iloc[:,1], drugcomb_drugs], axis=0, ignore_index=True).drop_duplicates()
# identifiers = pd.Series(['rocilinostat', 'saracatinib', 'ERKi', 'ERKi II', 'IGFR1', 'YL54' ])

mask = pd.Series(np.ones,index=identifiers, dtype="bool")
fgp = []
mol_desc = []
for i,x in enumerate(identifiers) :
    try:
        found = True
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + x + '/property/CanonicalSMILES/TXT'
        smiles = requests.get(url).text.rstrip()
        if('NotFound' in smiles):
            url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + x.replace("-","") + '/property/CanonicalSMILES/TXT'
            smiles = requests.get(url).text.rstrip()
            if('NotFound' in smiles):
                if re.search("\s",x):
                    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + x[:re.search("\s", x).start()] + '/property/CanonicalSMILES/TXT'
                    smiles = requests.get(url).text.rstrip()
                if('NotFound' in smiles):
                    print("not found", x)
                    mask.loc[x] = False
                    found=False

        if found: 
            if y := re.search("\n",smiles):smiles = smiles[:y.start()]
            if mol:=Chem.MolFromSmiles(smiles):
                print("found", x)
                fgp.append(list(fpgen.GetFingerprint(mol))+list(MACCSkeys.GenMACCSKeys(mol)))
                mol_desc.append(CalcMolDescriptors(mol))
    except Exception as e: 
        print("boo ", x, e)

# pd.concat([pd.Series(identifiers, name="drug_name"),pd.DataFrame(fgp),pd.DataFrame(mol_desc)], axis=1).to_pickle("data/drug_data.pkl.compress", compression="gzip")
export = pd.DataFrame(
    pd.concat([
        pd.DataFrame(fgp),
        pd.DataFrame(mol_desc)
    ], axis=1)
)

export.dropna(axis=1, inplace=True)
cols = export.columns.astype("str")
scaler = StandardScaler()
export = scaler.fit_transform(export.rename(columns=lambda colu:str(colu)))
print(mask.sum(), "out of", len(mask))
export = pd.DataFrame(export, index=pd.Series(identifiers[mask.to_numpy()], name="drug_name", dtype="category")).apply(pd.to_numeric, downcast="float")
export.to_pickle("data/drug_data.pkl.compress", compression="gzip")
# export = pd.read_pickle("data/drug_data.pkl.compress", compression="gzip")
# with open("scripts/preprocessing/drug_feat_dtype.txt", mode="w") as f: [f.write(str(typ)+"\n") for typ in export.dtypes]