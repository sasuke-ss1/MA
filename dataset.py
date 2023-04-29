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
    
        img = cv2.imread(path)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(labelDict[name], dtype=torch.float32)

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

        label =torch.tensor(labelDict[name], dtype=torch.float32)

        return label[[2, 4, 6, 7, 8]], label[7:]     

class DualImageVector(Dataset):
    def __init__(self, path: str, transform=None, labelDict=labelDict) -> None:
        super().__init__()

        self.imgPaths = [os.path.join(path, i) for i in sorted(os.listdir(path))]
        self.labelDict = labelDict
        self.transform = transform

    def __len__(self) -> int:
        return len(self.imgPaths)//2
    
    def __getitem__(self, index):
        path0, path1 = self.imgPaths[2*index], self.imgPaths[2*index + 1]
        
        name0 = path0.split("/")[-1].split(".")[0]
        name1 = path1.split("/")[-1].split(".")[0]

        assert name0[:-2] == name1[:-2], f"Images Not Paired, Found IMG0 to be {name0} and IMG1 to be {name1}" 

        img0, img1 = cv2.imread(path0), cv2.imread(path1)
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        label = torch.tensor(labelDict[name0[:-2]], dtype=torch.float32)

        return (img0.to(torch.float32), img1.to(torch.float32), label[[2, 4, 6, 7 ,8]]), label[7:]
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((224, 224))])
    data = DualImageVector("./NEW_Images", transform)
    (img0, img1, feat), y = data[1]

    print(img0.shape)
    print(img1.shape)
    print(feat.shape)
    