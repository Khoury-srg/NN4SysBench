import random

from model import model

import pandas as pd
import torch

from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

random.seed(2024)
save_path = "./result"
pt_path = 'pre-model-dict.pt'
pt_path = 'model-dict-4.pt'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


def predict(model, data_loader):
    preds = []

    all_labels = []

    model.eval()

    for batch in tqdm(data_loader, desc="Validation"):
        inputs, labels = batch

        outputs = torch.squeeze(model(inputs, train=False)).detach().numpy()
        all_labels.append(torch.squeeze(labels).detach().numpy())
        preds.append(outputs.round())
    preds = np.concatenate(preds).astype(int)
    all_labels = np.concatenate(all_labels).astype(int)
    return preds, all_labels


no_crime = pd.read_csv("no_crime_loc.csv")
no_crime.insert(no_crime.shape[1], 'label', 0)
crime = pd.read_csv("crime_processed.csv")
crime.insert(crime.shape[1], 'label', 0)
model = model()
sampled_crime = resample(crime, n_samples=no_crime.shape[0], random_state=2024)
train_data = pd.concat([sampled_crime, no_crime])
train_dataset = TensorDataset(torch.Tensor(train_data[['Lat', 'Long']].values),
                              torch.Tensor(train_data[['label']].values))

train_dataset, eval_dataset = random_split(train_dataset, [0.9, 0.1], generator=torch.Generator().manual_seed(2024))

val_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=10240)


def accuracy(pred, label):
    total = 0
    correct = 0
    number = len(pred)
    for i in range(number):
        if pred[i] == label[i]:
            correct += 1
        total += 1
    print(f"Accuracy is {correct} / {total}  {correct / total}")


model.eval()

model.load_state_dict(
    torch.load(os.path.join(save_path, pt_path)))
size = 20000
eval_data = train_data.sample(n=size)

x = eval_data['Lat'].values
y = eval_data['Long'].values
label = eval_data['label'].values
correct = 0
X1 = []
Y1 = []
X2 = []
Y2 = []
for i in range(size):
    input = torch.tensor([x[i], y[i]]).float()
    output = model(input, train=False).item()

    if output < 0.5:
        output = 0
        X2.append(x[i])
        Y2.append(y[i])

    else:
        output = 1
        X1.append(x[i])
        Y1.append(y[i])
    if output == label[i]:
        correct += 1
print(correct / size)
plt.scatter(x=X1, y=Y1)
# plt.scatter(x=X2, y=Y2)
plt.show()
# pos_acc = sum(preds[:crime_count] == all_labels[:crime_count]) / crime_count
# neg_acc = sum(preds[crime_count:] == all_labels[crime_count:]) / (data.shape[0] - crime_count)
# print(f"Accuracy is {acc}, accuracy for positive samples is {pos_acc}, for negative samples is {neg_acc}")
