import torch
from argparse import ArgumentParser
from torchvision import transforms
from dataset import ImageVector
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(10)
parser = ArgumentParser()

parser.add_argument("--pathImg", "-pi", default="./ALL_Images", type=str)
parser.add_argument("--batch_size", "-bs", default=32, type=int)
parser.add_argument("--epochs", "-e", default=20, type=int)
parser.add_argument("--learning_rate", "-lr", default=1e-3, type=float)
parser.add_argument("--modelPath", "-mP", default="./model.pth", type=str)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),transforms.Resize((224, 224))])

vecData = ImageVector(args.pathImg, transform)

model = torch.load(args.modelPath)
weights = model.resnet.conv1.weight

class VisualModel(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.conv1.weight = weights

    def forward(self, img):
        output = self.conv1(img)

        return output


img = vecData[0][0][0].unsqueeze(0)

visual = VisualModel(weights)

output = visual(img).squeeze(0).detach().numpy()

fig, axs = plt.subplots(8, 4)

for i in range(output.shape[0]):
    axs[i//4, i%4].imshow(output[i], cmap="gray")
    axs[i//4, i%4].set_xticks([])
    axs[i//4, i%4].set_yticks([])

plt.show()

















