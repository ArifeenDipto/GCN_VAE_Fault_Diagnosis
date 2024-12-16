# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:30:43 2024

@author: MA11201

Variational Graph Convolutional Autoencoder training file

"""

import sys
sys.path.append(r'C:\a_PhD Research\FDD\IPPT Exp\AE')
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch import optim
import seaborn as sns
import matplotlib.pyplot as plt
from AE_Model import AE
import warnings
warnings.filterwarnings("ignore")
data_folder = r"C:\a_PhD Research\FDD\IPPT Exp\IPPT_Processed_data\wind_size_5" #data folder

#reading data files from the folder
data = []
for file in sorted(os.listdir(data_folder)):
    data_file = pd.read_csv(os.path.join(data_folder, file))
    data_file = data_file.drop(['Unnamed: 0'], axis=1)
    data.append(data_file)

#scaling the data files using Standard Scaler
scaler = StandardScaler()
scaler.fit(data[0])
data0 = scaler.transform(data[0])
data0 = pd.DataFrame(data0)
data0.shape

data1 = data0.copy()
data1.columns = data[0].columns
data1.head(3)


X_tr = torch.tensor(np.array(data0.iloc[:7000, :]), dtype=torch.float)
X_tr.shape

#model hyper-parameters
model = AE(13)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
epoch =3000
reconstruction_function = nn.MSELoss()  # mse loss


#Model training process
tr_loss = []
for i in range(epoch):
    optimizer.zero_grad()
    X_pr = model(X_tr)
    loss = reconstruction_function(X_pr, X_tr)
    loss.backward()
    optimizer.step()
    tr_loss.append(loss.item())
    print("Epoch %d | Loss: %.4f" %(i, loss))

#train loss plot
plt.plot(np.array(tr_loss))
loss_tr = np.array(tr_loss)
np.savetxt(r"C:\a_PhD Research\FDD\Codes\AE\save_dir\AE_loss.txt", loss_tr)

#saving the model
torch.save(model.state_dict(), r"C:\a_PhD Research\FDD\IPPT Exp\AE\save_dir\AE_wind_5.pkl")



model_val = model.eval()

val_data = data1.iloc[7000:9000, :]
X_val = torch.tensor(np.array(val_data), dtype=torch.float)

val_pr = model_val(X_val)
val_pred1 = pd.DataFrame(val_pr.detach().numpy())
val_data1 = pd.DataFrame(X_val.detach().numpy())

plt.plot(val_data1.iloc[:, 0])
plt.plot(val_pred1.iloc[:, 0])


plt.plot(val_data1.iloc[:, 1])
plt.plot(val_pred1.iloc[:, 1])

plt.plot(val_data1.iloc[:, 2])
plt.plot(val_pred1.iloc[:, 2])

plt.plot(val_data1.iloc[:, 3])
plt.plot(val_pred1.iloc[:, 3])

plt.plot(val_data1.iloc[:, 4])
plt.plot(val_pred1.iloc[:, 4])

plt.plot(val_data1.iloc[:, 5])
plt.plot(val_pred1.iloc[:, 5])

plt.plot(val_data1.iloc[:, 6])
plt.plot(val_pred1.iloc[:, 6])

plt.plot(val_data1.iloc[:, 7])
plt.plot(val_pred1.iloc[:, 7])


plt.plot(val_data1.iloc[:, 8])
plt.plot(val_pred1.iloc[:, 8])

plt.plot(val_data1.iloc[:, 9])
plt.plot(val_pred1.iloc[:, 9])

plt.plot(val_data1.iloc[:, 10])
plt.plot(val_pred1.iloc[:, 10])

plt.plot(val_data1.iloc[:, 11])
plt.plot(val_pred1.iloc[:, 11])

plt.plot(val_data1.iloc[:, 12])
plt.plot(val_pred1.iloc[:, 12])


test_data = data1.iloc[10000:14000, :]
X_val = torch.tensor(np.array(val_data), dtype=torch.float)

val_pr = model_val(X_val)

val_pred1 = pd.DataFrame(val_pr.detach().numpy())
val_data1 = pd.DataFrame(X_val.detach().numpy())





