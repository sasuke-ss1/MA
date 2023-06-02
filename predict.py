import torch
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
import pandas as pd

model = torch.load("Model.h5")
print(model)
model.eval()

trainData = OnlyVector("Data4.txt", normalize=True)
testData = TestData("Data6.txt", trainData.mean, trainData.std)
testLoader = DataLoader(testData, batch_size=1)

ys = torch.zeros(testData.__len__(), dtype=torch.float32)
with torch.no_grad():
    for idx, x in tqdm(enumerate(testLoader)):
        ys[idx] = model(x)

df = pd.DataFrame(ys.numpy(), columns=["Predictions"])

df.to_csv("preds.csv")

