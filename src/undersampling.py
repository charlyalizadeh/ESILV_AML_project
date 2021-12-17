import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import numpy as np


data = pd.read_csv('../data/csv/train.csv')
data['temp_id'] = list(range(len(data.index)))
data = data[list(data.drop('transactionRevenue', axis=1).columns) + ['transactionRevenue']]
columns = data.columns
rus = RandomUnderSampler(random_state=42)
X = data.drop('transactionRevenue', axis=1).to_numpy()
y = (data['transactionRevenue'] != 0).astype(int).to_numpy()
true_y = data['transactionRevenue']
print(f'Dataset distribution {Counter(y)}')
X_res, y_res = rus.fit_resample(X, y)
print(f'Resampled distribution {Counter(y_res)}')
data_numpy = np.concatenate((X_res, y_res[None, :].T), axis=1)
data = pd.DataFrame(data=data_numpy, columns=columns)
data['transactionRevenue'] = true_y[data['temp_id']].tolist()
data.drop('temp_id', axis=1, inplace=True)
data.to_csv('../data/csv/train_rus.csv', index=False)
