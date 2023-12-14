# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLPptorchSigmoid
from neural import MLPptorchReLU
import torch
import torch.nn as nn

# функция обучения
def train(x, y, num_iter):
    for i in range(0,num_iter):
        pred = net.forward(x)
        loss = lossFn(pred, y)
        loss.backward()
        optimizer.step()
        if i%(num_iter/10)==0:
           print('Ошибка на ' + str(i) + ' итерации: ', loss.item())
    return loss.item()


# функция обучения
def train1(x, y, num_iter):
    for i in range(0,num_iter):
        pred = net1.forward(x)
        loss1 = lossFn1(pred, y)
        loss1.backward()
        optimizer1.step()
        if i%(num_iter/10)==0:
           print('Ошибка на ' + str(i) + ' итерации: ', loss1.item())
    return loss1.item()

df = pd.read_csv('data.csv')
df = df.iloc[np.random.permutation(len(df))]

X = df.iloc[0:100, 0:3].values
y = df.iloc[0:100, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)

X_test = df.iloc[100:150, 0:3].values
y = df.iloc[100:150, 4]
y = y.map({'Iris-setosa': 1, 'Iris-virginica': 2, 'Iris-versicolor':3}).values.reshape(-1,1)
Y_test = np.zeros((y.shape[0], np.unique(y).shape[0]))
for i in np.unique(y):
    Y_test[:,i-1] = np.where(y == i, 1, 0).reshape(1,-1)


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
#hiddenSizes = 50 # задаем число нейронов скрытого слоя 
hiddenSizes = [50, 20, 10]
outputSize = Y.shape[1] if len(Y.shape) else 1 # количество выходных сигналов равно количеству классов задачи

print("Sigmoid:")

net = MLPptorchSigmoid(inputSize,hiddenSizes,outputSize)
lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# как работает регуляризация
# optimizer = torch.optim.SGD(net.parameters(), lr=0.05, weight_decay=0.1)

loss_ = train(torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 5000)

# for name, param in net.named_parameters():
#     print(name, param)

pred = net.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y))
print(err)   

pred = net.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err = sum(abs((pred>0.5)-Y_test))
print(err)


acc1 = (len(Y_test)-sum(err))/len(Y_test) * 100



print("Точность :")
print(acc1)


print("\nReLU:")
##################################
net1 = MLPptorchReLU(inputSize,hiddenSizes,outputSize)
lossFn1 = nn.MSELoss()

optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.0001)

# как работает регуляризация
# optimizer = torch.optim.SGD(net.parameters(), lr=0.05, weight_decay=0.1)

loss_1 = train1(torch.from_numpy(X.astype(np.float32)), 
              torch.from_numpy(Y.astype(np.float32)), 5000)

# for name, param in net.named_parameters():
#     print(name, param)

pred1 = net1.forward(torch.from_numpy(X.astype(np.float32))).detach().numpy()
err1 = sum(abs((pred1>0.5)-Y))

print(err1)   

pred1 = net1.forward(torch.from_numpy(X_test.astype(np.float32))).detach().numpy()
err1 = sum(abs((pred1>0.5)-Y_test))
acc1 = (len(Y_test)-sum(err1))/len(Y_test) * 100
print(err1)
print("Точность:")
print(acc1)

