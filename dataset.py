import os
import cv2
import sys
from torch.utils.data import Dataset
import torch
from torch.nn.functional import normalize
from torchvision import transforms


path = "./Data.txt"
pathImg = "./ALL_Images"
with open(path, "r") as f:
    a = f.readlines()

labelDict = {}

for line in a:
    l = eval(line)
    name = l[0]
    data = l[1:]
    labelDict[name] = data


## Preprocessing code

class ImageVector(Dataset):
    def __init__(self, path: str, transform=None, labelDict=labelDict) -> None:
        super().__init__()

        self.imgPaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.labelDict = labelDict
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgPaths)
    
    def __getitem__(self, index):
        path = self.imgPaths[index]
        name = path.split("/")[-1].split(".")[0]
    
        img = cv2.imread(path)/255
        if self.transform:
            img = self.transform(img)

        label = normalize(torch.tensor(labelDict[name], dtype=torch.float32), p=1, dim=0)

        return (img.to(torch.float32), label[[2, 4, 6, 7 ,8]]), label[7:]


class OnlyVector(Dataset):
    def __init__(self, path: str,labelDict=labelDict) -> None:
        super().__init__()

        self.imgPaths = [os.path.join(path, i) for i in os.listdir(path)]
        self.labelDict = labelDict

    def __len__(self) -> int:
        assert len(self.labelDict) == len(self.imgPaths)
        return len(self.imgPaths)

    def __getitem__(self, index) -> torch.Tensor:
        path = self.imgPaths[index]
        name = path.split("/")[-1].split(".")[0]

        label = normalize(torch.tensor(labelDict[name], dtype=torch.float32), p=1, dim=0)

        return label[[2, 4, 6, 7, 8]], label[7:]     

if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    data = ImageVector(pathImg, transform)
    
    print(data[0][0][0].shape, data[0][0][1].shape)