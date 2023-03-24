import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import utils
from models import CNN, LSTMNN
from constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train(train_loader, model, optimizer, criterion):
    model=model.train()
    losses = []
    correct=0
    incorrect=0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += torch.sum(output.argmax(axis=1) == target)
        incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def test(test_loader, model, criterion):
    model=model.eval()
    losses = []
    correct = 0
    incorrect=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            correct += torch.sum(output.argmax(axis=1) == target)
            incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def load_data():
    X,y=utils.preproc_data(*utils.load_data())
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    train_loader=DataLoader(
        TensorDataset(torch.tensor(X_train,dtype=torch.long),torch.tensor(y_train,dtype=torch.long)),
        batch_size=TRAIN_BATCH_SIZE
    )
    test_loader=DataLoader(
        TensorDataset(torch.tensor(X_test,dtype=torch.long),torch.tensor(y_test,dtype=torch.long)),
        batch_size=TEST_BATCH_SIZE,
    )
    return train_loader,test_loader

def plt_losses(train_losses,test_losses,epochs):
    plt.figure()
    plt.plot(range(epochs),train_losses, label="Train Loss")
    plt.plot(range(epochs),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def train_test(model,train_loader,test_loader,optimizer,criterion):
    train_losses=[]
    test_losses=[]
    for epoch in range(EPOCHS):
        start=time.time()
        train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(train_loss)
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(test_loss)
        end=time.time()
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}, Epoch Time: {end-start}s')
    plt_losses(train_losses,test_losses,EPOCHS)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main(model_type):
    model = model_type()
    model.to(device)
    print(f'Number of parameters: {count_parameters(model)}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    train_test(model,*load_data(),optimizer,criterion)

if __name__ == "__main__":
    main(CNN)
    main(LSTMNN)