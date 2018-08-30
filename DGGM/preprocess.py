#!usr/bin/python

import os

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys

def retrieve_smiles(l):
    """ Retrieves the smiles representation from a line in file """
    
    l = str(l)
    l = l.split("\\t")
    return [l[0].split("'")[1], l[1].split("\\n")[0]]

def get_smiles_from_file(f):
    """ Gets SMILES from a file 'f' """

    lines = f.readlines()

    # Create SMILES list
    smiles = []
    for i in range(len(lines)):
        smiles.append(retrieve_smiles(lines[i]))

    # Remove all empty strings
    smiles = list(filter(None, [smiles[i][1] for i in range(len(smiles))]))
    
    return smiles

smiles = []
files = os.listdir('data3')
for i in tqdm(range(len(files))):
#     print ("Reading " + files[i] + "...")
    f = open('data3/' + files[i], 'rb')
    smiles += get_smiles_from_file(f)
    f.close()

print (len(smiles))