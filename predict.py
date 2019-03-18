from PIL import Image
from torch import nn,optim
from torchvision import datasets,transforms,models

import json
import torch
import numpy as np

'''
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_units, drop_p=0.5):
        
        super().__init__()
        # Input to a hidden layer
        self.hidden_units = nn.ModuleList([nn.Linear(input_size, hidden_units[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_units[:-1], hidden_units[1:])
        self.hidden_units.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_units[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        
        for each in self.hidden_units:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)
'''
    
def get_input_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',help='image_data',type=str)
    parser.add_argument('--checkpoint',help='checkpoint',type=str)
    parser.add_argument('--category_names',help='category_names',type=str)
    parser.add_argument('--topk',help='topk',type=int)
    parser.add_argument('--gpu',help='gpu',type=str)
    
    return parser.parse_args()

def load_data(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(filepath):
    checkpoint= torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    pil_img = Image.open(image)
    # Determine which demension is smaller
    # and resize to 256 while preserving aspect   
    if pil_img.size[0] > pil_img.size[1]: 
        pil_img.thumbnail((256 + 1, 256)) 
    else:
        pil_img.thumbnail((256, 256 + 1))            
    # Define the boundary and crop it around the center
    left = (pil_img.width - 224) / 2
    bottom = (pil_img.height - 224) / 2
    right = left + 224
    top = bottom + 224
    pil_img = pil_img.crop((left, bottom, right, top))
    # Normalize the img with np
    np_img = np.array(pil_img)
    np_img = np_img / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    np_img = np_img.transpose((2, 0, 1)) 
    np_img = torch.FloatTensor(np_img)
    
    return np_img

def predict(image_path, model, topk, gpu_mode):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Process the img                                                                                                                                                                                     
    img = process_image(image_path)
    if gpu_mode:
        model.cuda()
        img = img.cuda()
    # Make sure model in evaluation mode                                                                                                                                                                  
    model.eval()
    # Feed forward to get the prediction                                                                                                                                                                  
    with torch.no_grad():
        outputs = model.forward(img.unsqueeze(0))
    # Calc probabilities and classes                                                                                                                                                                      
    probs, class_indices = outputs.topk(topk)
    probs = probs.exp().cpu().numpy()[0]
    class_indices = class_indices.cpu().numpy()[0]
    # Convert indices to classes                                                                                                                                                                          
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in class_indices]
      
    return probs, classes

def main():
    in_args = get_input_args()
    input_data = in_args.input_data
    checkpoint = in_args.checkpoint
    category_names = in_args.category_names
    topk = in_args.topk
    gpu = in_args.gpu
    
    model = load_checkpoint(checkpoint)
    cat_to_name = load_data(category_names)
    probs,classes = predict(input_data,model,topk,gpu)
    print("K corresponding probs: ",probs)
    print("K corresponding classes: ",classes)
    
if __name__ == "__main__":
    main()
    