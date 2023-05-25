from torch.utils.data import Dataset
import torch


class OnlyVector(Dataset):
    def __init__(self, path: str, normalize: bool) -> None:
        super().__init__()
        
        with open(path, "r") as f:
            a = f.readlines()
        
        self.data = torch.zeros((len(a), len(eval(a[0]))), dtype=torch.float32)

        for i, line in enumerate(a):
            l = eval(line)
            self.data[i] = torch.tensor(l, dtype=torch.float32)
        
        if normalize:
            self.normalize()            
    
    def normalize(self) -> None:
        for i in range(self.data.shape[1] - 1):
            self.data[..., i] = (self.data[..., i] - torch.mean(self.data[..., i]))/torch.std(self.data[..., i])

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx) -> tuple:

        return self.data[idx, :7], self.data[idx, 7]     
    
if __name__ == "__main__":

    data = OnlyVector('./Data4.txt', False)
    print(data[0])
    