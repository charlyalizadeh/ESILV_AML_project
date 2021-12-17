import torch
from datamanager import DataManager
from sklearn.metrics import r2_score
import torch.nn as nn
from joblib import load
import numpy as np

# Load Neural Network
ffnn = nn.Sequential(
        nn.Linear(33, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
        nn.ReLU(),
)
ffnn.load_state_dict(torch.load('../model/ffnn_rus'))


# Load Random Forest
clf = load('../model/random_forest.joblib')


def predict(X):
    y_pred_binary = clf.predict(X.to_numpy())
    y_pred = ffnn(torch.tensor(X.to_numpy()).float())
    y_pred[np.argwhere(y_pred_binary == 0)] = 0
    return y_pred


def evaluate(X, y):
    y_pred = predict(X).detach()
    return r2_score(y, y_pred)


dataset = DataManager('../data/csv')
r2_train = evaluate(dataset.X_train, dataset.y_train)
r2_val = evaluate(dataset.X_val, dataset.y_val)
r2_test = evaluate(dataset.X_test, dataset.y_test)

print("R2 scores:")
print(f"  train: {r2_train}")
print(f"  test: {r2_test}")
print(f"  val: {r2_val}")
