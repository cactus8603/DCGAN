from audioop import bias
import torch.nn as nn

class D(nn.Module):
    def __init__(self, inputsize, hiddensize):
        super(D, self).__init__()
        
        self.main = nn.Sequential(
            # layer 1
            nn.Conv2d(inputsize, hiddensize, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 2
            nn.Conv2d(hiddensize, hiddensize*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 3
            nn.Conv2d(hiddensize*2, hiddensize*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize*4), 
            nn.LeakyReLU(0.2, inplace=True), 
            
            # layer 4
            nn.Conv2d(hiddensize*4, hiddensize*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hiddensize*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # layer 5
            nn.Conv2d(hiddensize*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def  forward(self, input):
        return self.main(input)
        