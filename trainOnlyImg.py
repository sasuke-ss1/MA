import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Model import *
from dataset import *
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split


parser = ArgumentParser()

parser.add_argument("--pathImg", "-pi", default="./ALL_Images", type=str)
parser.add_argument("--batch_size", "-bs", default=32, type=int)
parser.add_argument("--epochs", "-e", default=20, type=int)
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

vecData = ImageVector(args.pathImg, transform)
trainVec, testVec = train_test_split(vecData, test_size=0.2)
trainVecLoader, testVecLoader = DataLoader(trainVec, batch_size=args.batch_size, shuffle=True), DataLoader(testVec)

modelVec = ResNet_18(3, 2).to(device)
lossFn = nn.MSELoss()
optim = Adam(modelVec.parameters(), args.learning_rate)

def train():
    print("Training Begins")

    for e in range(1, args.epochs+1):
        for (img, x), y in tqdm(trainVecLoader):
            img, x, y = img.to(device), x.to(device), y.to(device)
            
            y_pred = modelVec(img)
            loss = lossFn(y_pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            loss = []
            for (img, x), y in testVecLoader:
                img, x, y = img.to(device), x.to(device), y.to(device)
                
                y_pred = modelVec(img)
                loss.append(lossFn(y_pred, y).item())

            print("The validation Loss in this Epoch is ",sum(loss)/len(loss), "\n")

if __name__ == "__main__":
    train()