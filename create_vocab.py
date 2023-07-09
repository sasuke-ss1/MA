import sys
from rdkit import Chem
import re
import pandas as pd

def canonicalize_smiles_from_files(path: str):
    data = pd.read_excel(path)
    smiles = list(set(data['Smiles'].tolist()))
    ret = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            ret.append(Chem.MolToSmiles(mol))
        except:
            pass
        
    return ret

def construct_vocabulary(smiles_list: list):
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]
    
    with open('data/voc', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def write_smiles_to_file(smiles_list, fname):
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_files(smiles_file)
    print('Constructing vocabulary...')
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")


