import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('data.xlsx')
smiles1 = data['Smiles'].tolist()


A = data['log10_H2_Bayesian'].tolist()
B=data['log10_O2_Bayesian'].tolist()
C=data['log10_N2_Bayesian'].tolist()
D=data['log10_CO2_Bayesian'].tolist()
E=data['log10_CH4_Bayesian'].tolist()


print(A[0])
for i in range(len(A)):
    A[i] = 10**A[i]
    B[i] = 10**B[i]
    C[i] = 10**C[i]
    D[i] = 10**D[i]
    E[i] = 10**E[i]


print(A[0])


#print(X1)




smiles = set(smiles1)

smiles = list(smiles)

print(smiles)

print(len(A), len(smiles1))

Y1=[]
Y2=[]
Y3=[]
Y4=[]
Y5=[]

for i in range(len(smiles)):
   
   num=0
   y1 = 0
   y2 = 0
   y3 =0
   y4=0
   y5=0
   for j in range(len(smiles1)):
      if(smiles[i]==smiles1[j]):
         y1=y1+A[j]
         y2=y2+B[j]
         y3=y3+C[j]
         y4=y4+D[j]
         y5=y5+E[j]
         num=num+1
         
 
   Y1.append(y1/num)
   Y2.append(y2/num)
   Y3.append(y3/num)
   Y4.append(y4/num)
   Y5.append(y5/num)


print(len(Y1), len(Y2), len(Y3), len(Y4), len(Y5))



# Convert SMILES to molecular representations (fingerprints)
mol_objects = [Chem.MolFromSmiles(smile) for smile in smiles]
fingerprints = [Chem.RDKFingerprint(mol) for mol in mol_objects]


print(fingerprints[0])

# Get the input shape based on the fingerprint length
input_shape = fingerprints[0].GetNumBits()
print(input_shape)

L=fingerprints[0].ToBitString()
# Convert strings to list of integers
L = [int(bit) for bit in L]

print(L)
