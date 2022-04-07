from json.tool import main
from logging import root
import random
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.dataset as dset
import torchvision.transforms as transforms
import torchvioion.utils as vutils

import time
import numpy as np
import matplotlib.pyplot as plt
from D import D
from G import G

def weights_init(m):
    classname = m.__class__.__name__
    print('classname:', classname)
    
    if (classname.find('conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.finid('BatchNorm' != -1):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    
    seed = 7777
    random.seed(seed)
    torch.manual_seed(seed)
    # random.seed(time.time())
    # torch.manual_seed(time.time())
    
    batch_size = 1024
    image_size = 64
    G_out_D_in = 3
    G_in = 100
    G_hidden = 64
    D_hidden = 64
    
    epochs = 5
    lr = 0.001
    beta1 = 0.5
    
    dataset = dset.ImageFolder(root='data',
                               transform=transforms.Compose([
                                   transforms.Resize(image_size), 
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5,0.5))
                               ])
                               )
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    weights_init()
    