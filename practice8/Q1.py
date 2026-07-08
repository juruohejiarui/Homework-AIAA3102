import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10

class Model(nn.Module) :
    class ResnetBlk(nn.Module) :
        def __init__(self, in_channels : int, out_channels : int, stride : int) :
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            if stride != 1 or in_channels != out_channels :
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else :
                self.shortcut = nn.Identity()
                
            self.relu = nn.ReLU()
        
        def forward(self, x : torch.Tensor) -> torch.Tensor :
            res : torch.Tensor = self.shortcut(x)
            
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            
            return self.relu(x + res)
    
    class InceptionModule(nn.Module) :
        def __init__(self, in_channels : int, out_channels : int) :
            super().__init__()
        
    def __init__(self, ) :
        super(Model, self).__init__()
        