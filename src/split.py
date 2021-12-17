#!/usr/bin/python

import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np

filepath = sys.argv[1]
suffix = sys.argv[2]


data = pd.read_csv(filepath)
data = data[list(data.drop('transactionRevenue', axis=1).columns) + ['transactionRevenue']]
columns = data.columns

# Shuffle before spliting
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Train/Test/Val = 80/10/10
X = data.drop('transactionRevenue', axis=1).to_numpy()
y = data['transactionRevenue'].to_numpy()
stratify = (data['transactionRevenue'] != 0).astype(int).tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
stratify = y_test != 0
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.1, random_state=42, stratify=stratify)

train = pd.DataFrame(data=np.concatenate((X_train, y_train[None, :].T), axis=1), columns=columns)
test = pd.DataFrame(data=np.concatenate((X_test, y_test[None, :].T), axis=1), columns=columns)
val = pd.DataFrame(data=np.concatenate((X_val, y_val[None, :].T), axis=1), columns=columns)

# Scaling all features
features = train.drop('transactionRevenue', axis=1).columns
scaler = MinMaxScaler()
train.loc[:, features] = scaler.fit_transform(train.loc[:, features])
val.loc[:, features] = scaler.transform(val.loc[:, features])
test.loc[:, features] = scaler.transform(test.loc[:, features])

train.to_csv(f"../data/csv/train{suffix}.csv", index=False)
val.to_csv(f"../data/csv/val{suffix}.csv", index=False)
test.to_csv(f"../data/csv/test{suffix}.csv", index=False)
