from audioop import bias
from tkinter.tix import Tree
import torch.nn as nn

class G(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize) -> None:
        super(G, self).__init__()
        self.main = nn.Sequential(
            # layer 1
            nn.ConvTranspose2d(inputsize, hiddensize*8, 4, 1, 0, bias=False),
            nn.BacthNorm2d(hiddensize*8),
            nn.ReLU(True),
            
            # layer 2
            nn.ConvTranspose2d(hiddensize*8, hiddensize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize*4),
            nn.ReLU(True),
            
            # layer 3
            nn.ConvTanspose2d(hiddensize*4, hiddensize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize*2),
            nn.ReLU(True),
            
            # layer 4
            nn.ConTranspose2d(hiddensize*2, hiddensize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize),
            nn.ReLU(),
            
            nn.ConvTranspose2d(hiddensize, outputsize, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)