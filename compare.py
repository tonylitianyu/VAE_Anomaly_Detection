import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAE2 import VAE
from VAE_ETH import VAE_ETH
import time
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from PIL import Image
import glob
import random


image_size = (192,256)
image_channel = 1
batch_size = 10
number_epoch = 30000
data_file_path = 'data'
latent_size = 20
curr_epoch = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# def preprocessInput(input):
#     input = np.array(input).reshape(image_size[0],image_size[1])
#     input = np.interp(input, (input.min(), input.max()), (0, 1))
#     input = torch.from_numpy(input)
#     input = input.float()
#     input = input.to(device)
#     return input


def preprocessInputForMine(input):
    files_data = []
    input = np.array(input).reshape(image_size[0],image_size[1])
    input = np.interp(input, (input.min(), input.max()), (0, 1))
    for i in range(0,10):
        files_data.append(input)
    inputs = np.array(files_data)
    inputs = torch.from_numpy(inputs)
    inputs = inputs.float()
    inputs = inputs.to(device)
    return inputs


# checkpoint_other = torch.load('model/ETH_1.pth')
# model_other = checkpoint_other['model']
# normal_other = preprocessInput(Image.open('test/normal.jpg').convert('L'))
# human_other = preprocessInput(Image.open('test/human.jpg').convert('L'))


checkpoint_mine = torch.load('model/AL.pth')
model_mine = checkpoint_mine['model']
normal_mine = preprocessInputForMine(Image.open('test/normal.jpg').convert('L'))
human_mine = preprocessInputForMine(Image.open('test/human.jpg').convert('L'))



# def generateModelTestResult(model,model_name,input,input_name):
#     model_other.eval()
#     output_batch = model_other(input)
#     loss = VAE_ETH.lossFunction(output_batch,input)
#     print("----------------------------------------------------------------------")
#     print("Generating result of the " + input_name + " image using " + model_name)
#     print("Loss for " + model_name + " is " + str(loss.item()))
#     print("----------------------------------------------------------------------")
#     save_image(output_batch.cpu(), 'test/'+ model_name + '_' + input_name + '.png')

def generateModelTestResultForMine(model,model_name,input,input_name):
    model_mine.eval()
    output_batch, mean, logstd = model_mine(input)
    loss = VAE.lossFunction(output_batch[:1],input[:1],mean,logstd)
    print("----------------------------------------------------------------------")
    print("Generating result of the " + input_name + " image using " + model_name)
    print("Loss for " + model_name + " is " + str(loss.item()))
    print("----------------------------------------------------------------------")

    save_image(output_batch.view(-1, 1, image_size[0], image_size[1])[:1].cpu(), 'test/'+ model_name + '_' + input_name + '.png')







# generateModelTestResult(model_other,'ETH',human_other,'human')
# generateModelTestResult(model_other,'ETH',normal_other,'normal')



generateModelTestResultForMine(model_mine,'Mine',human_mine,'human')
generateModelTestResultForMine(model_mine,'Mine',normal_mine,'normal')
