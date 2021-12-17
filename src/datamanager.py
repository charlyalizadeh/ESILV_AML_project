import pandas as pd
from pathlib import PurePath


class DataManager:
    def __init__(self, datapath, rus=False):
        if rus:
            self.train = pd.read_csv(PurePath(datapath, 'train_rus.csv'))
        else:
            self.train = pd.read_csv(PurePath(datapath, 'train.csv'))
        self.val = pd.read_csv(PurePath(datapath, 'val.csv'))
        self.test = pd.read_csv(PurePath(datapath, 'test.csv'))
        self.X_train, self.y_train = self.train.drop('transactionRevenue', axis=1), self.train['transactionRevenue']
        self.X_val, self.y_val = self.val.drop('transactionRevenue', axis=1), self.val['transactionRevenue']
        self.X_test, self.y_test = self.test.drop('transactionRevenue', axis=1), self.test['transactionRevenue']
        self.y_train_binary = (self.y_train != 0).astype(int)
        self.y_val_binary = (self.y_val != 0).astype(int)
        self.y_test_binary = (self.y_test != 0).astype(int)
