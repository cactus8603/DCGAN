from json.tool import main
from logging import root
import random
import torch.nn as nn
import torch.optim as opt
import torch.utils.data
import torchvision.dataset as dset
import torchvision.transforms as transforms
import torchvioion.utils as vutils

import time
import numpy as np
import matplotlib.pyplot as plt
from D import D
from G import G

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def weights_init(m):
    classname = m.__class__.__name__
    print('classname:', classname)
    
    if (classname.find('conv') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.finid('BatchNorm' != -1):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(dataLoader):
    batch_size = 1024
    image_size = 64
    G_out_D_in = 3
    G_in = 100
    G_hidden = 64
    D_hidden = 64
    
    epochs = 5
    lr = 0.001
    beta1 = 0.5
    
    netG = G(G_in, G_hidden, G_out_D_in).to(device)
    netG.apply(weights_init)
    
    netD = D(G_out_D_in, D_hidden).to(device)
    netD.apply(weights_init)
    
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, G_in, 1, 1, device=device)
    real_label = 1
    fake_label = 0
    opt_D = opt.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_G = opt.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    img_list = []
    G_loss = []
    D_loss = []
    
    iters = 0
    
    for epoch in range(epochs):
        for i, data in enumerate(dataLoader, 0):
            # update D
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = netD(real_cpu).view(-1)
            
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            noise = torch.randn(b_size, G_in, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            
            errD_fake = criterion(output, label)
            errD_fake.backward()
            
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            opt_D.step()
            
            # update G
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            opt_G.step()
            
            if (i % 50 == 0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' % (epoch, epochs, i, len(dataLoader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # save loss
            G_loss.append(errG.item())
            D_loss.append(errD.item())
            
            if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(dataLoader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
    torch.save(netD, 'netD.pkl')
    torch.save(netG, 'netG.pkl')
        
    return G_loss, D_loss, img_list 
            
def plotImg(G_loss, D_loss, img_list):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss during Training")
    plt.plot(G_loss, label="G")
    plt.plot(D_loss, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataLoader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    
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
    
    G_loss, D_loss, img_list = train(dataLoader)
    
    plotImg(G_loss, D_loss, img_list)
    
    