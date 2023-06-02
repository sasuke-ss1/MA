from dataset import OnlyVector
from Model import FNN
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import MultiStepLR
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
device = torch.device("cpu") 

parser = ArgumentParser()
parser.add_argument("--path", "-p", default="./Data4.txt", type=str, help="Path of the data.txt file")
parser.add_argument("--normalize", "-n", default=1, type=int, help="Flag for normalizing data, 0 means False")
parser.add_argument("--hiddenSize", "-hs", default=128, type=int, help="The size of the hidden layers")
parser.add_argument("--numLayers", "-nl", default=3, type=int, help="The Number of Hidden Layers")
parser.add_argument("--activation", "-a", default="Mish", type=str, help="The Activation function used for the hidden layers")
parser.add_argument("--valSize", "-ts", default=0.25, type=float, help="The size of the validation set in fraction of the total data")
parser.add_argument("--batchSize", "-bs", default=64, type=int, help="The batch size for the data")
parser.add_argument("--epochs", "-e", default=300, type=int, help="Number of epochs to train for")
parser.add_argument("--learningRate", "-lr", default=1e-3, type=float, help="Initial Learning Rate")
parser.add_argument("--org", "-o", type=str, default="const", help="Organization per layer")
parser.add_argument("--name", "-name", type=str, default="1", help="Name of the plot")

args = parser.parse_args();args.normalize = bool(args.normalize)

data = OnlyVector(args.path, args.normalize)
trainData, valData = train_test_split(data, test_size=args.valSize, random_state=RANDOM_SEED)

trainLoader = DataLoader(trainData, batch_size=args.batchSize, shuffle=True)
valLoader = DataLoader(valData, batch_size=args.batchSize)

model = FNN(7, args.hiddenSize, args.numLayers,args.activation).to(device)
print(model)

lossFn = MSELoss()
optim = Adam(model.parameters(), lr=args.learningRate)
scheduler = MultiStepLR(optim, [100, 200], gamma=0.5)

for e in range(args.epochs):
    loop_obj = tqdm(trainLoader)
    trainLoss = []
    model.train()
    for x, y in loop_obj:
        x, y = x.to(device), y.to(device)

        optim.zero_grad()
        pred = model(x)
  
        loss = lossFn(pred, y)

        loss.backward()
        optim.step()

        loop_obj.set_description(f"Epoch: {e+1}")
        loop_obj.set_postfix({"Loss": loss.item()})
        trainLoss.append(loss.item())
    print(f"Average Train Loss: {sqrt(sum(trainLoss)/len(trainLoss))}")

    valLoss = []

    with torch.no_grad():
        model.eval()
        for x, y in valLoader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            valLoss.append(lossFn(pred, y))

        print(f"Average Validation Loss: {sqrt(sum(valLoss)/len(valLoss))}")
    print("\n")
    scheduler.step()

# Plotting first layer weights

weights = list(model.nn.children())[0].weight.detach().numpy()
sample = next(iter(trainLoader))[0].numpy()

plot = np.mean(sample@weights, axis=0)[..., np.newaxis]

im = plt.matshow(plot, cmap="RdGy")
plt.colorbar(im)
plt.xticks([])
plt.savefig(args.name+".png")
plt.show()

torch.save(model, "Model.h5")
