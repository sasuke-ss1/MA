from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class OnlyVector(Dataset):
    def __init__(self, path: str, normalize: bool) -> None:
        super().__init__()
        with open(path, "r") as f:
            a = f.readlines()
        
        self.data = torch.zeros((len(a), len(eval(a[0]))), dtype=torch.float32)

        for i, line in enumerate(a):
            l = eval(line)
            self.data[i] = torch.tensor(l, dtype=torch.float32)
        
        self.mean = torch.mean(self.data, dim=0)
        self.std = torch.std(self.data, dim=0)

        if normalize:
            self.normalize()            
    
    def normalize(self) -> None:
        for i in range(self.data.shape[1] - 1):
            self.data[..., i] = (self.data[..., i] - torch.mean(self.data[..., i]))/torch.std(self.data[..., i])

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple:

        return self.data[idx, :7], self.data[idx, 7], self.data[idx, 8]     
    
class TestData(Dataset):
    def __init__(self, path: str, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        with open(path, "r") as f:
            a = f.readlines()
        
        self.data = torch.zeros((len(a), 7), dtype=torch.float32)
        
        for i, line in enumerate(a):
            nums = [float(i) for i in line[1:-2].split(",")[1:-1]]
            self.data[i] = torch.tensor(nums, dtype=torch.float32)

        self.data = (self.data - mean[..., :-1])/std[..., :-1]

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx]


def plot(path: str):
    with open(path, "r") as f:
        a = f.readlines()

    data = np.zeros((len(a), len(eval(a[0]))), dtype=np.float32)

    for i, line in enumerate(a):
        l = eval(line)
        data[i] = np.array(l, dtype=np.float32)
    df = pd.DataFrame(data)
    
    for i in range(0, 9):
        plt.figure()
        df[i].plot(kind="hist")
    
    plt.show()
if __name__ == "__main__":
    plot("./Data_10_5.txt")
    #data = OnlyVector('./Data_10_5.txt', True)
   
    