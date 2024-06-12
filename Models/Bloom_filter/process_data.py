import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

random.seed(2024)

size = 3000
grid = np.zeros([size, size])

crime_df = pd.read_csv('crime_processed.csv')[['Lat', 'Long']]
crime_df.insert(crime_df.shape[1], 'label', 1)

lats = crime_df['Lat'].values
longs = crime_df['Long'].values


def is_ocp(lat, long):
    x = lat * size
    y = long * size
    if x == size or y == size:
        return True
    if grid[int(x)][int(y)] == 1:
        return True
    else:
        grid[int(x)][int(y)] = 1
        return False


index = 0
for lat, long in tqdm(zip(lats, longs), total=len(lats)):
    is_ocp(lat, long)

X = []
Y = []
X1 = []
Y1 = []
for i in range(size):
    for j in range(size):
        if grid[i][j] == 0:
            X.append(i)
            Y.append(j)
        else:
            X1.append(i)
            Y1.append(j)

plt.scatter(x=X1, y=Y1)
# plt.scatter(x=X, y=Y)

plt.show()


def is_ocp2(lat, long):
    x = lat * size
    y = long * size
    if grid[int(x)][int(y)] == 1:
        return True
    else:
        return False


def generate_random_gps():
    latitude = float(random.uniform(0, 1))
    longitude = float(random.uniform(0, 1))
    loga = '%.8f' % longitude
    lata = '%.8f' % latitude
    if not is_ocp2(float(lata), float(loga)):
        return f'({float(lata)}, {float(loga)})', loga, lata
    else:
        return False


df_size = crime_df.shape[0] * 10
gene_logs = list(range(df_size))

gene_lats = list(range(df_size))
index = 0
with tqdm(total=df_size) as pbar:
    while index < df_size:
        returned = generate_random_gps()
        if returned is False:  # or returned[0] in locations:
            continue
        else:

            gene_loc, log, lat = returned
            # locations.append(gene_loc)
            gene_logs[index] = float(log)
            gene_lats[index] = float(lat)
            index = index + 1
            pbar.update(1)

mydf = pd.DataFrame({'Long': gene_logs, 'Lat': gene_lats})
mydf.drop_duplicates()
mydf.to_csv('no_crime_loc.csv', index=False)

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

number = 100000
crime_df = crime_df.sample(n=number)
lats = crime_df['Lat'].values
longs = crime_df['Long'].values

lats = np.array(lats)
longs = np.array(longs)
plt.scatter(x=lats, y=longs)

mydf = mydf.sample(n=number)
lats = mydf['Lat'].values
longs = mydf['Long'].values

lats = np.array(lats)
longs = np.array(longs)

# plt.scatter(x=lats, y=longs)
plt.show()
