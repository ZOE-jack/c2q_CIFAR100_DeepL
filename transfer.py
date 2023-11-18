#Import package for system, time calculation, and ploting tool.
import numpy as np
import time
import os,sys
from tempfile import TemporaryDirectory
import matplotlib.pyplot as plt
import json
#Import essential torch packages 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
#Import self-defined quantum net
from qnet import Quantumnet
loss = {"train":[], "val":[]}
acc = {"train":[], "val":[]}
print(torch.cuda.is_available())
#Input command
q_choice=True#bool(int(sys.argv[1]))
batch_size=64#int(sys.argv[2])
filtered_classes = ["bear", "tiger"]#[sys.argv[3], sys.argv[4]]

#Select operating device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Data preprocessing 
transformer_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(256, antialias=True),
     transforms.CenterCrop(224),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
     #do normalizing over RGB

transformer_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize(224, antialias=True),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
     #do normalizing over RGB


trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True,
                                        download=True,transform=transformer_train)

testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False,
                                       download=True,transform=transformer_test)

#Filtered data for target classification
classes = trainset.classes
#['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
filtered_labels=[classes.index(cl) for cl in filtered_classes]
print(filtered_labels)
sub_indices={'train': [], 'val': []}#store filtered data for train, validation data
image_datasets_full={'train': trainset, 'val': testset}

#Find the ordered index of goal classes in datasets
for phase in ['train', 'val']:
    for idx, label in enumerate(image_datasets_full[phase].targets):
        if label in filtered_labels:
            sub_indices[phase].append(idx)

#split our data by ordered target index for train, validation data
image_datasets = {x: Subset(image_datasets_full[x], sub_indices[x])
                    for x in ['train', 'val']}

#Make batches for iteration training for train, validation data
dataloaders = {x: DataLoader(image_datasets[x],#Shuffle make data mess up 
                    batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

#data size for train, validation data
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

print(dataset_sizes)
#Maps CIFAR labels to the index of filtered_labels
def labels_to_filtered(labels):
    return [filtered_labels.index(label) for label in labels]

#Train our model
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    begin_train = time.time()
    best_acc = 0.0

    #Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        
        for epoch in range(num_epochs):
            print()
            #Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  #Training mode:open batch normalization and dropout
                else:
                    model.eval()   #Evaluate mode:disable batch normalization and dropout

                running_loss = 0.0
                running_corrects = 0
                iter = 1#record current iteration 
                n_batches = dataset_sizes[phase] // batch_size
                
                #Iterate over data
                for inputs, labels in dataloaders[phase]:
                    begin_batch = time.time()#record batch time costing
                    batch_size_ = len(inputs)#just varify batch size
                    inputs = inputs.to(device)
                    labels = torch.tensor(labels_to_filtered(labels))
                    labels = labels.to(device)

                    #Initiate the parameter gradients(optional)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):#Renew parameter only in train mode 
                        outputs = model(inputs)#Forward propagation
                        _, preds = torch.max(outputs, dim=1)#the biggest probability for single row(image)
                        loss = criterion(outputs, labels)#loss 

                        # backward + optimize only if in training mode
                        if phase == 'train':
                            loss.backward()#back propagation according to loss to calculate gradient
                            optimizer.step()#update parameter
                    #Update current conditions
                    batch_time = time.time() - begin_batch
                    print(f'Phase: {phase} Epoch: {epoch + 1}/{num_epochs} Iter: {iter}/{n_batches + 1} Batch time: {batch_time:.4f}', end='\r')
                    
                    iter += 1
                    #Statistics for info
                    running_loss += loss.item() * batch_size_#Binary_crossentropy had been divided by batches
                    running_corrects += torch.sum(preds == labels.data)
                    batch_corrects = torch.sum(preds == labels.data).item()
                    
                if phase == 'train':#Renew learning rate after an epoch of training 
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print(f'Epoch{epoch+1}/{num_epochs}-->{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}                        ', end="\r")
                print()


                #Find the best accuracy between epoches
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
                    
            print()

        time_elapsed = time.time() - begin_train
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        # save model, best weights and load best model weights
        torch.save(model, "model3.pt")
        model.load_state_dict(torch.load(best_model_params_path))
        if q_choice:
            torch.save(model.state_dict(), 
                'quantum_' + filtered_classes[0] + '_' + filtered_classes[1] + '3.pt'
            )
        else:
            torch.save(model.state_dict(), 
                'classical_' + filtered_classes[0] + '_' + filtered_classes[1] + '3.pt'
            )
    return model

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model_outputs(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {filtered_classes[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)

def train():
    model_ft = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model_ft.fc.in_features
    ## to freeze the layers
    for param in model_ft.parameters():
        param.requires_grad = False

    if q_choice:
        model_ft.fc = Quantumnet(len(filtered_classes), device, num_ftrs)

    else:model_ft.fc = nn.Linear(num_ftrs, len(filtered_classes))
    model_ft = model_ft.to(device)

    #for name, param in model_ft.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=15)
    visualize_model_outputs(model_ft)
    
    plt.ioff()
    plt.show()
if __name__ =='__main__':
    print('training and testing starts...')
    train()