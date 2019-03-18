import numpy as np
import torch
import torch.nn.functional as F

from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import datasets,transforms,models


def get_input_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir',help='train_data_dir',type=str)
    parser.add_argument('--valid_data_dir',help='valid_data_dir',type=str)
    parser.add_argument('--test_data_dir',help='test_data_dir',type=str)
    parser.add_argument('--arch',help='arch',type=str)
    parser.add_argument('--learning_rate',help='learning rate',type=float)
    parser.add_argument('--hidden_units',help='hidden',type=int)
    parser.add_argument('--epochs',help='epochs',type=int)
    parser.add_argument('--gpu',help='gpu',type=str)
    
    return parser.parse_args()


def load_data(train_dir,valid_dir,test_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])

    valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                                [0.229,0.224,0.225])])
    
    test_transforms = transforms.Compose([transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir,transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset,batch_size=64,shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=True)
    
    return trainloader,validloader,testloader,train_dataset


def Definition_mode(lr):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential(OrderedDict([
                              ('fc1',nn.Linear(25088,4096)),
                              ('relu1',nn.ReLU()),
                              ('dropout1',nn.Dropout(p=0.5)),
                              ('fc2',nn.Linear(4096,1000)),
                              ('relu2',nn.ReLU()),
                              ('dropout2',nn.Dropout()),
                              ('fc3',nn.Linear(1000,102)),
                              ('output',nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(),lr=lr)
    return model,criterion,optimizer,classifier

def train(model, trainloader, validloader, criterion, optimizer, epochs, 
          device,log_interval = 20):
    steps = 0
    running_loss = 0
    model.train()
    model.to('cuda')
    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % log_interval == 0: 
                # Make sure network is in eval mode for inference
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, valid_accuracy = validate(model, validloader, criterion)
                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / log_interval),
                          "Valid Loss: {:.3f}.. ".format(valid_loss / len(validloader)),
                          "Valid Accuracy: {:.3f}".format(valid_accuracy / len(validloader)))
                    running_loss = 0
                    running_accu = 0
                    # Make sure training is back on
                    model.train()

def validate(model, validloader, criterion, device = 'cuda'):
    loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)
        # Log loss and accuracy on validation data
        output = model.forward(images)
        loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return loss, accuracy

def save_model(arch,model,input_size,classifier,optimizer,train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'arch': arch,
                  'class_to_idx': model.class_to_idx,
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_units': [4096,1000],
                  'epochs': 3,
                  'log_interval': 32,
                  'learning_rate': 0.001,
                  'classifier': classifier,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()}

    torch.save(checkpoint,'checkpoint.pth')
    
def main():
    in_args = get_input_args()
    train_data_dir = in_args.train_data_dir
    valid_data_dir = in_args.valid_data_dir
    test_data_dir = in_args.test_data_dir
    arch = in_args.arch
    lr = in_args.learning_rate
    hidden_units = in_args.hidden_units
    epochs = in_args.epochs
    gpu = in_args.gpu
    
    trainloader,validloader,testloader,train_dataset = load_data(train_data_dir,valid_data_dir,test_data_dir)
    model,criterion,optimizer,classifier = Definition_mode(lr)
    train(model,trainloader,validloader,criterion,optimizer,epochs,device=gpu)
    save_model(arch,model,25088,classifier,optimizer,train_dataset)
    
if __name__ == "__main__":
    main()
    
    