# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 22:47:28 2020

@author: KJP
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn import datasets, model_selection

from matplotlib import pyplot as plt
from matplotlib import cm

import pandas as pd

import urllib.request



from scipy.io import loadmat


# 데이터 정규화
mnist_data = mnist['data'] / 255

pd.DataFrame(mnist_data)

plt.imshow(mnist_data[0].reshape(28, 28), cmap=cm.gray_r)
plt.show()
mnist_label = mnist['target']
mnist_label
train_size = 50000
test_size = 500
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data, 
                                                                    mnist_label, 
                                                                    train_size=train_size, 
                                                                    test_size=test_size
                                                                   )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.get_device_name(0)


train_X = torch.from_numpy(train_X).float().to(device)
train_Y = torch.from_numpy(train_Y).long().to(device)


test_X = torch.from_numpy(test_X).float().to(device)
test_Y = torch.from_numpy(test_Y).long().to(device)

print(train_X.shape)
print(train_Y.shape)
train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=100, shuffle=True)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(784, 256)
    self.fc2 = nn.Linear(256, 256)
    self.fc3 = nn.Linear(256, 256)
    self.fc4 = nn.Linear(256, 128)
    self.fc5 = nn.Linear(128, 128)
    self.fc6 = nn.Linear(128, 10)
    
    
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.dropout(x, training=self.training)
    x = self.fc6(x)
    return F.log_softmax(x)
    
model = Net()
model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
  
  total_loss = 0
  
  for train_x, train_y in train_loader:
    
    train_x, train_y = Variable(train_x), Variable(train_y)
    
    optimizer.zero_grad()
    
    output = model(train_x)
    
    
    loss = criterion(output, train_y)
    
    loss.backward()
    
    optimizer.step()
    
    total_loss += loss.data.item()
    
  if (epoch+1) % 100 == 0:
    print(epoch+1, total_loss)
    
    
    
test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.cpu().data.numpy() == result.cpu().numpy()) / len(test_y.cpu().data.numpy())

accuracy




