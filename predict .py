import argparse
import json
from PIL import Image
import torch
import numpy as np
from train import builder,is_gpu
from torchvision import models,transforms
import matplotlib.pyplot as plt
import os
import math
os.environ['QT_QPA_PLATFORM']='offscreen'

def arg_parser():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--top_k',type=int)
    parser.add_argument('--gpu',type =int)
    parser.add_argument('--category_names')
    args = parser.parse_args()
    
    return args


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    print(checkpoint['name'])
    
    model,__,__ = builder(checkpoint['name'],25088,102,4096,checkpoint['learn_rate'])
        
    for param in model.parameters(): 
        param.requires_grad = False
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    
    return model




def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    transform = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    im = Image.open(image)
    
    preprocessed_image = transform(im)
    return preprocessed_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model,cat2json,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if topk == None:
        topk = 5
    if cat2json ==None:
        cat2json = 'cat_to_name.json' 
    with open(cat2json, 'r') as f:
        cat_to_name = json.load(f)
    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze_(0)
    
    model.eval
    output = model.forward(image)
    ps = torch.exp(output)
    
    ps_top = ps.topk(topk)
    idxtoclass = {value: key for key, value in model.class_to_idx.items()}
    
    probs = ps_top[0].tolist()[0]
    classes = [idxtoclass[i] for i in ps_top[1].tolist()[0]]
    
    #return category
    cat = []
    for i in classes:
        cat.append(cat_to_name[str(i)])
        
    return probs, cat


# TODO: Display an image along with the top 5 classes
def sanity_check(prob,classes,path):
    
    for i,j in zip(prob,classes):
        print("The Probability is of Category {} is {:.2f}".format(j,i))
        
    
    
#     fig = plt.figure()
#     ax = fig.add_axes([0,0,1,1])
#     ax.barh(classes,prob)
#     img = process_image(path)
#     imshow(img)
#     plt.show()


if __name__ == '__main__':
    args = arg_parser()
    checkpoint = args.checkpoint
    image = args.image
    top_k = args.top_k
    model = load_checkpoint(checkpoint)
    device = is_gpu(args.gpu)
    model.to(device)
    cat2json = args.category_names
    
    if checkpoint == None:
        model = load_checkpoint('checkpoint.pth')
    if image == None:
        image = process_image('flowers/test/1/image_06743.jpg')   
    
    prob,classes = predict(image,model,cat2json,top_k)
    
    sanity_check(prob,classes,image)
    
    
    
    
    
    
    