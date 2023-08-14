# import rdkit


# drug['fps']=drug['smiles'].apply(lambda x: list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x))) if Chem.MolFromSmiles(x) is not None else '')


from urllib.request import urlopen
from urllib.parse import quote
import pandas as pd
import requests
import re
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import rdkit.Chem as Chem
from rdkit.Chem.Descriptors import CalcMolDescriptors

fpgen = AllChem.GetMorganGenerator(radius=3)
oneil = pd.read_csv("data_raw/oneil.csv", usecols=["drugA_name", "drugB_name"])
yosua = pd.read_csv("data/test_yosua.csv", usecols=["drugA_name", "drugB_name"])
identifiers = pd.unique(pd.concat([oneil.iloc[:,0],oneil.iloc[:,1],yosua.iloc[:,0],yosua.iloc[:,1]]))
# identifiers = ['rocilinostat', 'saracatinib']
fgp = []
mol_desc = []
for i,x in enumerate(identifiers) :
    try:
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/' + x + '/property/CanonicalSMILES/TXT'
        smiles = requests.get(url).text.rstrip()

        if('NotFound' in smiles):
            print(x, " not found")
        else: 
            if x := re.search("\n",smiles):smiles = smiles[:x.start()]
            if mol:=Chem.MolFromSmiles(smiles):

                fgp.append(list(fpgen.GetFingerprint(mol))+list(MACCSkeys.GenMACCSKeys(mol)))
                mol_desc.append(CalcMolDescriptors(mol))
    except Exception as e: 
        print("boo ", x, e)

pd.concat([pd.Series(identifiers, name="drug_name"),pd.DataFrame(fgp),pd.DataFrame(mol_desc)], axis=1).to_pickle("data/drug_data.pkl.compress", compression="gzip")