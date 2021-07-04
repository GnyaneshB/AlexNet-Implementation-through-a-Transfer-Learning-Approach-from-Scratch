# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:31:56 2021

@author: cgnya
"""
########Generic Modules########
import torch
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import os
import copy



########User-Defined Modules########
from AlexNet import AlexNet

class ANTL:
    
    def __init__(self):
        
        self.data_transforms = {
                                'train' : transforms.Compose([
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),
                                'test' : transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
                                }
        self.data_dir = 'C:/Users/cgnya/OneDrive/Desktop/MY LIBRARY/Books/IT/Pytorch/data/cifar10'
        self.image_datasets = {x : datasets.ImageFolder(os.path.join(self.data_dir, x),
                                           self.data_transforms[x])
                                    for x in ['train', 'test']}
        self.dataloaders = {x : torch.utils.data.DataLoader(self.image_datasets[x], batch_size=256,
                                                shuffle=True)
                                    for x in ['train', 'test']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'test']}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = 40
        self.criterion = torch.nn.CrossEntropyLoss()
        print(self.device)
        self.class_names = self.image_datasets['train'].classes

        
    def train(self, model, criterion, optimizer):
        since = time.time()
    
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(self.epochs):
            print('Epochs {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train() # set model to training mode
                else:
                    model.eval() # set model to validation mode
                
                running_loss = 0.0
                running_corrects = 0
                #Iterate over data
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0) 
                    running_corrects += torch.sum(preds == labels.data)
       
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print('{} loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                       
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed%60))
        print('Best Test acc : {}'.format(best_acc))
        
        model.load_state_dict(best_model_wts)
        return model

    
    def pretrain(self):
        model_pt = AlexNet().to(self.device)
        op = optim.SGD(model_pt.parameters(), lr=0.01, momentum=0.9)
        pt = self.train(model_pt, self.criterion, op)
        self.visualize(pt)
        return pt
    
        
    def fine_tune(self):
        model_ft = self.pretrain()
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = torch.nn.Linear(num_ftrs, 10)
        model_ft = model_ft.to(self.device)
        op = optim.SGD(model_ft.classifier[6].parameters(), lr=0.01, momentum=0.9)
        ft = self.train(model_ft, self.criterion, op)
        self.visualize(ft)
        
        
    def visualize(self, model, num_images=4):
        def imshow(inp,title = None):
            inp = inp.numpy().transpose((1,2,0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1) # limits the values in an array btn range [0,1]
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.0001)
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()
        
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(inputs.to(self.device))
                _, preds = torch.max(outputs, 1)
                
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far) 
                    ax.axis('off')
                    ax.set_title('Predicted Label: {} , Ground Label: {}'.format(self.class_names[preds[j]], self.class_names[labels[j]]))
                    imshow(inputs.cpu().data[j])
                    
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return 
            model.train(mode=was_training)
