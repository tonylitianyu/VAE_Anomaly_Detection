import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional


class VAE(nn.Module):
    def __init__(self,input_size, input_c,latent_size, batch_size):
        super(VAE,self).__init__()
        self.input_size = input_size
        self.input_c = input_c

        self.latent_size = latent_size
        self.batch_size = batch_size

        self.size_after_conv = 48*64
        #encoder
        self.f1 = nn.Conv2d(input_c,64,kernel_size=4,padding=1,stride=2)
        #self.b1 = nn.BatchNorm2d(64)
        self.f2 = nn.Conv2d(64,128,kernel_size=4,padding=1,stride=2)
        #self.b2 = nn.BatchNorm2d(128)
        # self.f3 = nn.Conv2d(64,128,kernel_size=4,stride=2)
        # self.b3 = nn.BatchNorm2d(128)


        self.r1 = nn.ReLU()

        self.fc11 = nn.Linear(self.batch_size*self.size_after_conv,1024)
        self.fc21 = nn.Linear(1024,20)
        self.fc12 = nn.Linear(self.batch_size*self.size_after_conv,1024)
        self.fc22 = nn.Linear(1024,20)




        #decoder
        self.fc31 = nn.Linear(20, 1024)
        self.fc32 = nn.Linear(1024,self.batch_size*self.size_after_conv)
        self.d1 = nn.ConvTranspose2d(128,64,kernel_size=4,padding=1,stride=2)
        #self.b4 = nn.BatchNorm2d(64)
        # self.d2 = nn.ConvTranspose2d(64,32,kernel_size=4, stride=2)
        # self.b5 = nn.BatchNorm2d(32)
        self.d3 = nn.ConvTranspose2d(64,input_c,kernel_size=4,padding=1,stride=2)


        self.r2 = nn.ReLU()

        self.s1 = nn.Sigmoid()

    def encoder(self,input):
        input = input.view(-1,self.input_c,self.input_size[0],self.input_size[1])

        h1 = functional.relu(self.f1(input))

        h2 = functional.relu(self.f2(h1))

        h2 = h2.view(-1,self.batch_size*self.size_after_conv)

        mean = functional.relu(self.fc11(h2))
        mean = self.fc21(mean)
        #print(mean)
        logstd = functional.relu(self.fc12(h2))
        logstd = self.fc22(logstd)

        std = torch.exp(0.5*logstd)
        eps = torch.randn_like(std)
        latent = mean+eps*std

        return latent,mean,logstd


    def decoder(self,latent):
        h1 = functional.relu(self.fc31(latent))
        h1 = functional.relu(self.fc32(h1))

        h1 = h1.view(-1,128,48,64)

        h2 = functional.relu(self.d1(h1))
        #print(h2)


        h3 = torch.sigmoid(self.d3(h2))

        h3 = h3.view(-1,self.input_size[0]*self.input_size[1])

        return h3


    def forward(self,input):
        latent,mean,logstd = self.encoder(input)
        output = self.decoder(latent)
        return output,mean,logstd

    def lossFunction(output, input, mean, logstd):

        generationLoss = functional.binary_cross_entropy(output,input.view(-1,192*256), reduction='sum')
        latentLoss = -0.5*torch.sum(1+logstd-mean.pow(2)-logstd.exp())
        return generationLoss + latentLoss
