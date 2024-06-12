import random

from model import model

import pandas as pd
import torch
import numpy
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import os

random.seed(2024)
save_path = "./result"
epochs=100
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)

no_crime = pd.read_csv("no_crime_loc.csv")
no_crime.insert(no_crime.shape[1], 'label', 0)
crime = pd.read_csv("crime_processed.csv")
crime.insert(crime.shape[1], 'label', 1)
model = model()
sampled_crime = resample(crime, n_samples=no_crime.shape[0], random_state=2024)
train_data = pd.concat([sampled_crime, no_crime])
train_dataset = TensorDataset(torch.Tensor(train_data[['Lat', 'Long']].values), torch.Tensor(train_data[['label']].values))
train_data, eval_data = random_split(train_dataset, [0.99,0.01], generator=torch.Generator().manual_seed(2024))


train_dataloader = DataLoader(train_data, shuffle=True, batch_size=1024)
model.apply(weights_init)
loss_fn = BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10 * len(train_dataloader))


model.train()
for epoch in range(epochs):
    loss_total = 0.
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
        loss_item = 0.
        optimizer.zero_grad()
        inputs, labels = batch

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss_item += loss.item()
        loss_total += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        # with warmup_scheduler.dampening():

    lr_scheduler.step()
    print("Epoch {}, loss: {}".format(epoch + 1, loss_total))
    if epoch%1==0:
        torch.save(model.state_dict(), os.path.join(save_path, f'model-dict-{epoch}.pt'))


if not os.path.exists(save_path):
    os.makedirs(save_path)
