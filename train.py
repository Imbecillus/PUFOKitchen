# DEBUG
# Some variables that will later be given as command line parameters are set here
trainset_path = "alldata_train"
testset_path = "alldata_test"
learning_rate = 0.001
delta_abort = 0.05
acc_abort = 90
max_epochs = 100
model_savepath = "nicoroffel_model.pth"

print('PUFOKitchenNet booting up...')

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import helpers

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('No GPU found; using CPU')

transforms = [torchvision.transforms.Resize((256, 256))]

classes = [o for o in os.listdir(trainset_path)
           if os.path.isdir(os.path.join(trainset_path, o))]
print(f'Our classes are: {classes}')

# Initialize dataset and dataloader
print(f'Initializing training set from path "{trainset_path}"...')
from dataset import PUFOKitchen
dataset = PUFOKitchen(trainset_path, data_transforms=torchvision.transforms.Compose(transforms))
train_dl = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
print(f'  Total number of pictures: {len(dataset)}')

print(f'Initializing test set from path "{testset_path}"...')
testset = PUFOKitchen(testset_path, data_transforms=torchvision.transforms.Compose(transforms))
test_dl = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)
print(f'  Total number of pictures: {len(testset)}')

print('Initializing model...')
resnet = torchvision.models.resnet50(progress=True).to(device)
dense = torch.nn.Linear(1000, 256).to(device)
final = torch.nn.Linear(256, len(classes)).to(device)
model = torch.nn.Sequential(resnet, dense, final).to(device)

print('Starting training...')
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(f'  learning_rate={learning_rate}')

def loss_batch(model, loss_func, prediction, yb, opt=None):
    loss = loss_func(prediction, yb)        

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()

start = time.time()
epoch = 1
abort = False
previous_error = 1000

while not abort:
    epoch_time = time.time()
    print(f'{epoch}: ', end='')

    # Training
    model.train()
    train_losses = []

    for xb, yb in train_dl:
        prediction = model(xb.to(device))
        tl = loss_batch(model, loss_func, prediction.to(device), yb.to(device), opt)
        train_losses.append(tl)

    # Calculate average loss at epoch end and print it
    ll = 0
    ln = 0
    for l in train_losses:
        ll += l
        ln += 1
    training_loss = ll / ln
    print(f'{round(training_loss, 4)}; ', end='')

    # Evaluation
    model.eval()
    train_acc, confusion_matrix_train = helpers.eval(model, train_dl, device, len(classes))
    print(f'{round(train_acc, 2)}% Accuracy on train set. ', end='')
    
    test_acc, confusion_matrix_test = helpers.eval(model, test_dl, device, len(classes))
    print(f'{round(test_acc, 2)}% Accuracy on test set. ', end='')


    print(f'Took {helpers.time_since(epoch_time)}')

    # Check whether we should abort now
    delta = previous_error - training_loss
    if delta < 0:
        delta *= -1

    if delta < delta_abort:
        print(f'Aborting: Change in loss was lower than {delta_abort}.')
        abort = True
    
    if epoch == max_epochs:
        print(f'Aborting: Maximum number of epochs ({max_epochs}) has been reached.')
        abort = True

    if train_acc > acc_abort:
        print(f'Aborting: Training accuracy has risen above {acc_abort}.')
        abort = True

    epoch += 1

print(f'Printing and exporting confusion matrix for train set...')
helpers.print_confusion_matrix(confusion_matrix_train, classes, model_savepath[0:-4] + '_conf_train.png')

print(f'Printing and exporting confusion matrix for test set...')
helpers.print_confusion_matrix(confusion_matrix_test, classes, model_savepath[0:-4] + '_conf_test.png')

torch.save(model.state_dict(), model_savepath)
print(f'Model exported to {model_savepath}.')
print(f'Total training time: {helpers.time_since(start)}.')
print("We're done! :)")