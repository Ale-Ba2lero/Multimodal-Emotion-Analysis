import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from skimage import transform
from tqdm import tqdm

import random

from RAVDESS_dataset_util import ToTensor, CenterCrop, Rescale

class EmoClassCNN(nn.Module):
    def __init__(self, image_size, num_classes):
        super(EmoClassCNN, self).__init__()
        
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2))
        
        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2))
        
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.BatchNorm2d(32))
    
        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2))
        
        self.layer5 = torch.nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2))
        
        self.layer6 = torch.nn.Sequential(
            nn.Linear(32*(self.image_size//4)*(self.image_size//4), 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))
        
        self.layer7 = torch.nn.Sequential(
            nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))
        
        self.layer8 = torch.nn.Sequential(
            nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5))
        
        self.layer9 = torch.nn.Sequential(
            nn.Linear(128, self.num_classes))
        
        
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        output = output.view(-1, 32*(self.image_size//4)*(self.image_size//4))
        output = self.layer6(output)
        output = self.layer7(output)
        output = self.layer8(output)
        output = self.layer9(output)
        return output

    
def train(model, dataset_loader, loss_fn, optimizer, num_epochs=5):
    
    training_loss = []
    test_loss = []
    best_accuracy = 0.0

    trainset_loader = dataset_loader[0]
    testset_loader = dataset_loader[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        running_acc = 0.0
        total = 0.0

        for i, sample in enumerate(trainset_loader):
            images = Variable(sample['image'].to(device))
            labels = Variable(sample['cat'].to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total += labels.size(0)
            running_loss += loss.item()

        training_loss.append(running_loss * 100 / total)

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy, test_loss_epoch = testAccuracy(model, testset_loader, loss_fn)
        test_loss.append(test_loss_epoch)
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))

        # save the model if the accuracy is the best
        '''
        if accuracy > best_accuracy:
            path = "./emoclassmodel_01.pth"
            torch.save(model.state_dict(), path)
            best_accuracy = accuracy'''

    return training_loss, test_loss


def testAccuracy(model, testset_loader, loss_fn):
    model.eval()

    accuracy = 0.0
    total = 0.0
    running_loss = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for sample in testset_loader:
            images = Variable(sample['image'].to(device))
            labels = Variable(sample['cat'].to(device))

            outputs = model(images)
            running_loss += loss_fn(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy = (100 * accuracy / total)
    running_loss = (100 * running_loss / total)
    return(accuracy, running_loss) 


def testBatch(model, trainsetLoader, categories):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get batch of images from the test DataLoader  
    sample = next(iter(trainsetLoader))
    images = Variable(sample['image'].to(device))
    labels = Variable(sample['cat'].to(device))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    batch_size = len(labels)

    # Show the real labels on the screen 
    print('Real labels:', '\t'.join('%5s' % categories[labels[j].item()] for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted:', '\t'.join('%5s' % categories[predicted[j].item()] for j in range(batch_size)))

    
def predict(model, image, image_size, device=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
        
    toTensor = ToTensor()
    crop = CenterCrop(image_size)
    rescale = Rescale(image_size)
    
    image = toTensor(crop(rescale(image)))
    image = torch.reshape(image, (1, 3, image_size, image_size))
    image = Variable(image.to(device))
    
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Function to show the images
def imageshow(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()