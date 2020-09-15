
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
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import glob
import random

#initialize value
continueTraining = False
model_name = 'AL'
data_file_path = 'human/All2/*.jpg'
test_folder = 'results2'
image_size = (192,256)
image_channel = 1
batch_size = 15
number_epoch = 50

latent_size = 20
curr_epoch = 0




if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
    torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Running on the CPU")


files_jpg = glob.glob(data_file_path)

files_data = []
for i in range(0,len(files_jpg)):
    input = Image.open(files_jpg[i]).convert('L')#L for black and white
    input = np.array(input).reshape(image_size[0],image_size[1],image_channel)
    input = np.interp(input, (input.min(), input.max()), (0, 1))
    files_data.append(input)



files_data = np.array(files_data)[0:5240]
total_data = len(files_data)
print(total_data/batch_size)


model = VAE(image_size,image_channel,latent_size,batch_size).to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)



#plot ready
fig = plt.figure()
fig.show()
# fig.canvas.draw()
x_axis_data = []
y_axis_data = []
def update_line(x,y):
    x_axis_data.append(x)
    y_axis_data.append(y)
    plt.plot(x_axis_data,y_axis_data,'r-')
    # if len(x_axis_data) == 1:
    plt.xlim([0,number_epoch])
    #     plt.ylim([0,max(y_axis_data)])
    # else:
    #     plt.xlim([min(x_axis_data),max(x_axis_data)])
    #     plt.ylim([min(y_axis_data),max(y_axis_data)])
    fig.canvas.draw()
    fig.canvas.flush_events()








def transformImagefromPNGtoData(input):

    input = torch.from_numpy(input)

    input = input.float()
    input = input.to(device)
    return input



if continueTraining:
    checkpoint = torch.load('model/'+model_name+'.pth')
    curr_epoch = checkpoint['epoch']
    print(curr_epoch)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']
    image_channel = checkpoint['image_channel']
    batch_size = checkpoint['batch_size']
    number_epoch = checkpoint['number_epoch']
    x_axis_data = checkpoint['plot_x']
    y_axis_data = checkpoint['plot_y']

for epoch in range(number_epoch):
    epoch = epoch + curr_epoch
    print('Begin Running Epoch {}'.format(epoch))
    begin_time = time.time()
    #training
    model.train()
    train_loss = 0

    for k in range(0,int(total_data/batch_size)):
        indices = np.random.choice(files_data.shape[0], batch_size, replace=False)

        inputs = transformImagefromPNGtoData(files_data[indices])
        optimizer.zero_grad()
        output_batch, mean, logstd = model(inputs)

        loss = VAE.lossFunction(output_batch,inputs,mean,logstd)

        loss.backward() #backpropogate
        train_loss += loss.item()
        optimizer.step() #gradient descent
        torch.cuda.memory_allocated()
        torch.cuda.memory_cached()


    #testing
    model.eval()
    test_loss = 0
    i = 0
    print('testing')
    with torch.no_grad():
        for k in range(0,int(total_data/batch_size)):
            indices = np.random.choice(files_data.shape[0], batch_size, replace=False)
            inputs = transformImagefromPNGtoData(files_data[indices])
            output_batch, mean, logstd = model(inputs)
            loss = VAE.lossFunction(output_batch,inputs,mean,logstd)

            test_loss = test_loss + loss.item()
            if i == 0 and epoch % 5 == 0:
                save_image(inputs.view(batch_size,image_channel,image_size[0], image_size[1])[:1].cpu(), test_folder+'/input_navigation' + str(epoch) + '.png')
                #comparison = torch.cat([inputs, output_batch.view(1, 1, image_size[0], image_size[1])])
                save_image(output_batch.view(batch_size, image_channel,image_size[0], image_size[1])[:1].cpu(), test_folder+'/reconstruction_navigation' + str(epoch) + '.png')
                # singleloss = VAE.lossFunction(output_batch,inputs,mean,logstd)
                # print(singleloss)
                checkpoint = {
                    'epoch':epoch,
                    'model':model,
                    'optimizer':optimizer,
                    'image_channel':image_channel,
                    'batch_size':batch_size,
                    'number_epoch':number_epoch,
                    'plot_x':x_axis_data,
                    'plot_y':y_axis_data
                }
                torch.save(checkpoint, 'model/'+model_name+'.pth')


                i = i+1




    test_loss /= int(total_data/batch_size)
    print('Average Loss: {}'.format(test_loss))
    update_line(epoch, test_loss)


    end_time = time.time()
    print('End Running Epoch {}, Duration: {}'.format(epoch, end_time-begin_time))
