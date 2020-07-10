import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import json
import argparse
from collections import OrderedDict


def arg_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',help='The directory of the dataset')
    parser.add_argument('--save_dir',help='The directory for saving checkpoint.pth')
    parser.add_argument('--arch',type=str,help='The model architecture you want to trained on')
    parser.add_argument('--learning_rate',type =int,help='The learning rate for the model')
    parser.add_argument('--hidden_units',type=int)
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--print_every',type=int)
    parser.add_argument('--gpu',type=str)
    
    
    args = parser.parse_args()
    return args

def transformer(data_dir):
    
    if data_dir == None:
        data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])


    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

 
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    return trainloader,validloader,testloader,train_data
    

def builder(arch,inputsize,outputsize,hiddenlayer,learn_rate):
    if learn_rate == None:
        learn_rate = 0.001
    if arch == None:
        model = models.vgg11(pretrained=True)
        print('selecting vgg11 as a default model')
    if hiddenlayer==None:
        hiddenlayer = 4096
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        model.name = 'vgg11'
        inputsize = 25088
    else: 
        model = eval("models.{}(pretrained=True)".format(arch))
        print('selecting '+arch+' as a model to train')
    model.name = arch
    if arch == 'densenet121':
      inputsize = model.classifier.in_features
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(inputsize, hiddenlayer,bias=True)),
                              ('relu', nn.ReLU()),
                              ('dropout1',nn.Dropout(p=0.5)),

                                ('fc2', nn.Linear(hiddenlayer, 1048,bias=True)),
                              ('relu', nn.ReLU()),
                              ('dropout2',nn.Dropout(p=0.5)),

                              ('fc3', nn.Linear(1048, outputsize,bias=True)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr=learn_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model,optimizer,criterion



def trainer(epochs,print_every,model,trainloader,validloader):
    if print_every == None:
        print_every = 30
    if epochs == None:
        epochs = 2
    steps = 0
    print("Training process initializing .....\n")
    for e in range(epochs):
        running_loss = 0
        model.train() # Technically not necessary, setting this for good measure

        for inputss, labelss in trainloader:
            steps += 1

            inputss, labelss = inputss.to(device), labelss.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputss)
            loss = criterion(outputs, labelss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in (validloader):
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()




                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.3f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.3f} | ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                model.train()

    print("\nTraining process is now complete!!")
    return model
    
def save_checkpoint(model,train_data,save_dir):
    
    
    if save_dir == None:
        save_dir = os.getcwd()
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
                          'epoch':2,
                          'class_to_idx':model.class_to_idx,
                          'state_dict': model.state_dict(),
                          'learn_rate':0.001,
                          'classifier':model.classifier,
                          'name':model.name}

    torch.save(checkpoint, 'checkpoint.pth')
    print('The model is saved in file checkpoint.pth')
def is_gpu(gpu):
    if gpu==None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('selecting ' +str(device)+ ' for training')
        return device
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    print('selecting' + str(device)+ 'for training')
    return device
def test_validator(model,testloader):
    print('\nCalculating accuracy for test dataset \n')
    accuracy = 0
    total_test=0
    correct_test =0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            t_image, mask = data
            t_image, mask = t_image.to(device),mask.to(device)


            outputs = model(t_image) 
            _, predicted = torch.max(outputs.data, 1)
            total_test += mask.size(0)
            correct_test += predicted.eq(mask.data).sum().item()
            test_accuracy = 100 * correct_test / total_test                                  

    print("Testing Accuracy: %d %%" % (test_accuracy))
    print('\n')

if __name__ == '__main__':
    args = arg_parser()
    data_dir = args.data_dir
    if data_dir == None:
        data_dir = 'flowers'

    trainloader,validloader,testloader,train_data = transformer(data_dir)
    arch = args.arch
    
    hidden_units = args.hidden_units
    device = is_gpu(args.gpu)
    learn_rate = args.learning_rate
    model,optimizer,criterion = builder(arch,25088,102,hidden_units,learn_rate)
    model.to(device)
    
    epoch = args.epoch
    print_every = args.print_every
    
    model = trainer(epoch,print_every,model,trainloader,validloader)
    test_validator(model,testloader) 
    save_dir = args.save_dir
    save_checkpoint(model,train_data,save_dir)

