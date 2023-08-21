import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn import tree
from sklearn.metrics import mean_squared_error

simple_tree = tree.DecisionTreeRegressor()

data = pd.read_excel('data.xlsx', na_values=['?'])
smiles1 = data['Smiles'].tolist()


A = data['log10_H2_Bayesian'].tolist()
B=data['log10_O2_Bayesian'].tolist()
C=data['log10_N2_Bayesian'].tolist()
D=data['log10_CO2_Bayesian'].tolist()
E=data['log10_CH4_Bayesian'].tolist()



for i in range(len(A)):
    A[i] = 10**A[i]
    B[i] = 10**B[i]
    C[i] = 10**C[i]
    D[i] = 10**D[i]
    E[i] = 10**E[i]




smiles = set(smiles1)

smiles = list(smiles)

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


descriptors = list(np.array(Descriptors._descList)[:,0])
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptors)

def computeDescriptors(mol, calculator):
    res = np.array(calculator.CalcDescriptors(mol))
    if not np.all(np.isfinite(res)):
        return None  #make is easier to identify problematic molecules (.e.g infity descriptor values) later 
    return res


smiles = ['*C*']#, 'C1CCCC1', 'CCCCCC', 'CCCC(C)C', 'CCC(C)(C)C']
y = [-3.18, -2.64, -3.84, -3.74, -3.55]
# Convert SMILES to molecular representations (fingerprints)
mol_objects = [Chem.MolFromSmiles(smile) for smile in smiles]
fingerprints = [Chem.RDKFingerprint(mol) for mol in mol_objects]

features = [computeDescriptors(x, calculator) for x in mol_objects]
exit(0)
simple_tree.fit(features, y)
preds = simple_tree.predict(features)

print("Mse: ", mean_squared_error(y, preds))