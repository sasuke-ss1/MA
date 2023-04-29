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
from torch.optim.lr_scheduler import MultiStepLR

torch.manual_seed(10)

parser = ArgumentParser()

parser.add_argument("--pathImg", "-pi", default="./NEW_Images", type=str)
parser.add_argument("--batch_size", "-bs", default=32, type=int)
parser.add_argument("--epochs", "-e", default=20, type=int)
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((224, 224))])

vecData = DualImageVector(args.pathImg, transform)
trainVec, testVec = train_test_split(vecData, test_size=0.2)
trainVecLoader, testVecLoader = DataLoader(trainVec, batch_size=args.batch_size, shuffle=True), DataLoader(testVec)

modelVec = DualImgVector(3, 64, 5, 32,2).to(device)
lossFn = nn.MSELoss()
optim = Adam(modelVec.parameters(), args.learning_rate)
scheduler = MultiStepLR(optim, milestones=[5, 10, 20, 25],gamma=0.5)

def train():
    print("Training Begins")
    full_val = []
    for e in range(1, args.epochs+1):
        for (img0, img1, x), y in tqdm(trainVecLoader):
            img0, img1,x, y = img0.to(device), img1.to(device), x.to(device), y.to(device)
            
            y_pred = modelVec(img0, img1, x)
            loss = lossFn(y_pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
        
        scheduler.step()

        with torch.no_grad():
            loss = []
            for (img0, img1, x), y in testVecLoader:
                img0, img1, x, y = img0.to(device), img1.to(device),x.to(device), y.to(device)
                
                y_pred = modelVec(img0, img1, x)
                loss.append(lossFn(y_pred, y).item())

            print(f"The validation Loss in after Epoch{e} is {sum(loss)/len(loss):0.5f}", "\n")
            full_val.append(sum(loss)/len(loss))
    print(f"Final average loss {sum(full_val)/len(full_val)}")
if __name__ == "__main__":
    train()