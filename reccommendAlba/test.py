#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## 아래는 실제 트레인 과정 입니다.


# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import pandas as pd
from pandas import DataFrame, Series
import csv
from random import randrange


# In[2]:


def load_data_real():
    df = pd.read_csv('data.csv', error_bad_lines=False)

    data_mok = df['mok']
    data_hwa = df['hwa']
    data_to = df['to']
    data_gm = df['gm']
    data_su = df['su']
    data_E = df['E']
    data_I = df['I']
    data_S = df['S']
    data_N = df['N']
    data_T = df['T']
    data_F = df['F']
    data_J = df['J']
    data_P = df['P']
    data_job = df['job']
    
    length = len(data_mok)
    
    data = []
    job = []

    for i in range(0,length):
        row = []
        row.append(data_mok[i])
        row.append(data_hwa[i])
        row.append(data_to[i])
        row.append(data_gm[i])
        row.append(data_su[i])
        row.append(data_E[i])
        row.append(data_I[i])
        row.append(data_S[i])
        row.append(data_N[i])
        row.append(data_T[i])
        row.append(data_F[i])
        row.append(data_J[i])
        row.append(data_P[i])
        
        data.append(row)
        label = []
        label.append(data_job[i])
        job.append(label)
    
    return data, job, length


# In[3]:


class myDataset(Dataset):
    def __init__(self, features, label):
        self.features = features
        self.label = label
        self.len = len(self.label)
        self.label_list = list(sorted(set(self.label)))
    
    def __getitem__(self, index):
        return self.features[index], self.label[index]
    def __len__(self):
        return self.len
    def get_labels(self):
        return self.label_list
    def get_label(self, id):
        return self.label_list[id]
    def get_label_id(self,label):
        return self.label_list.index(label)


# In[4]:


BATCH_SIZE = 16
epochs = 50

data, label, length = load_data_real()

#label indexing
vocab = set()
vocab_label = np.array(label)
vocab.update(vocab_label.flatten())
label_vocab = {word:i for i, word in enumerate(vocab)}
print(label_vocab)

final_label = []
for index, label_ in enumerate(label):
    final_label.append([label_vocab[label_[0]]])
    
data = np.array(data, dtype=np.float32)
final_label = np.array(final_label, dtype=np.float64)
final_label = final_label.flatten()
data = Variable(torch.from_numpy(data))
final_label = Variable(torch.from_numpy(final_label))

train_data = data[:18000]
train_label = final_label[:18000]
test_data = data[18000:length]
test_label = final_label[18000:length]

train_dataset = myDataset(train_data, train_label)
test_dataset = myDataset(test_data, test_label)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                         shuffle=False)

# Training settings
# batch_size = 64

# # MNIST Dataset
# train_dataset = datasets.MNIST(root='./mnist_data/',
#                                train=True,
#                                transform=transforms.ToTensor(),
#                                download=True)

# test_dataset = datasets.MNIST(root='./mnist_data/',
#                               train=False,
#                               transform=transforms.ToTensor())

# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)
class myModel(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(myModel, self).__init__()
        self.l1 = nn.Linear(13,50)
        self.l2 = nn.Linear(50,30)
        self.l3 = nn.Linear(30,15)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = x.view(-1,13)
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
class Model(nn.Module):

    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.l1 = nn.Linear(8, 6)
        self.l2 = nn.Linear(6, 4)
        self.l3 = nn.Linear(4, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the data (n, 1, 28, 28)-> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)
# our model
model = myModel()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        target = torch.tensor(target,dtype=torch.long)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            
def my_test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        target = torch.tensor(target,dtype=torch.long)
        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max
        pred = output.data.max(1, keepdim=True)[1]
        target = target.view(pred.size(0),1)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
# Training loop
for epoch in range(epochs):
    train(epoch)
    my_test()
    


# In[ ]:


##아래는 데이터 만드는 과정입니다.


# In[5]:


torch.save(model,'saved_model')


# In[ ]:





# In[43]:


model2 = torch.load('saved_model')
myinput = [1,2,2,1,0,2,4,2,4,2,4,2,4]
myinput = np.array(myinput,dtype=np.float32)
myinput = Variable(torch.from_numpy(myinput))
output = model2(myinput)
# print(output)
output = output.view(-1)
list = [i[0] for i in sorted(enumerate(output), key=lambda x:x[1], reverse=True)]
# print(output)
pos_list = sorted(output, reverse=True)
# print(pos_list)
# print(list)
# print(label_vocab)
sum_ = sum(pos_list)
# print(sum_)
# print(label_vocab.items())

job_list_ = []
pos_list_ = []

print(pos_list)
for num in pos_list:
    pos_list_.append(num/sum_)

print(pos_list_)
for num in list:    
    a=[name for name, age in label_vocab.items() if age == num]
    job_list_.append(a[0])
    
print(job_list_)


# In[ ]:





# In[8]:


def load_data():
    df = pd.read_csv('data.csv', error_bad_lines=False)

    data_mok = df['mok']
    data_hwa = df['hwa']
    data_to = df['to']
    data_gm = df['gm']
    data_su = df['su']
    data_E = df['E']
    data_I = df['I']
    data_S = df['S']
    data_N = df['N']
    data_T = df['T']
    data_F = df['F']
    data_J = df['J']
    data_P = df['P']
    
    length = len(data_mok)
    
    data = []

    for i in range(0,length):
        row = []
        row.append(data_mok[i])
        row.append(data_hwa[i])
        row.append(data_to[i])
        row.append(data_gm[i])
        row.append(data_su[i])
        row.append(data_E[i])
        row.append(data_I[i])
        row.append(data_S[i])
        row.append(data_N[i])
        row.append(data_T[i])
        row.append(data_F[i])
        row.append(data_J[i])
        row.append(data_P[i])

        data.append(row)


    print(data[0])
    print(data[1])
    
    com = data[0]
    
    jobs = []
    
    for data_ in data:
        select_list = data_[:5]
        max_ = max(data_[:5])
        list_ = [ i for i, x in enumerate( select_list ) if x == max_ ]
        random_index = randrange(0,len(list_))
        
        last_num = list_[random_index]
        
        if last_num == 0:
            list__ = ['바리스타', '매장', '의류']
            random_index = randrange(0,len(list__))
            job = list__[random_index]
            jobs.append(job)
        elif last_num == 1:
            list__ = ['레스토랑', '사무보조', '미용']
            random_index = randrange(0,len(list__))
            job = list__[random_index]
            jobs.append(job)
        elif last_num == 2:
            list__ = ['생산', '주방', '회계']
            random_index = randrange(0,len(list__))
            job = list__[random_index]
            jobs.append(job)
        elif last_num == 3:
            list__ = ['유아', '스포츠', 'PC']
            random_index = randrange(0,len(list__))
            job = list__[random_index]
            jobs.append(job)
        elif last_num == 4:
            list__ = ['배달', '창고', '서빙']
            random_index = randrange(0,len(list__))
            job = list__[random_index]
            jobs.append(job)
    print(jobs)
    return jobs


# In[9]:



jobs = load_data()

data = DataFrame(jobs)
print(data)
data.to_excel('label_2.xlsx', sheet_name='sheet1')
# data.to_csv('label_2.csv', index=False, header=False, encoding='ms949')
# f = open('label.csv', 'w', encoding='euc_kr', newline='')
# wr = csv.writer(f)
# for job in jobs:
#     wr.writerow(job)


# In[ ]:




