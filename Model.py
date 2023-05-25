import torch
import torch.nn as nn


class FNN(nn.Module):
    def __init__(self, inDims: int, hiddenDims: int, numLayers: int, activation: str, org="const") -> None:
        super().__init__()
        
        self.activation = getattr(nn, activation)()
        dims = [inDims]

        if org == "const":
            dims += [hiddenDims]*numLayers
        elif org == "half":
            for i in range(numLayers):
                dims.append(int(hiddenDims/2**(i)))

        dims.append(1)  
        layers= [nn.Linear(inDims, inDims), nn.Tanh(), nn.LayerNorm(inDims)]
        
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))            
            layers.append(self.activation)
            layers.append(nn.LayerNorm(dims[i+1]))

        layers.append(nn.Linear(dims[-2], dims[-1]))

        self.nn = nn.Sequential(*layers)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.nn(x)
        
        return(out.squeeze(1))
    