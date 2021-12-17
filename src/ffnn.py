from datamanager import DataManager
import torch
from torch import tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, RMSprop
import matplotlib.pyplot as plt


def get_tensordataset(X_df, y_df):
    return TensorDataset(tensor(X_df.to_numpy()).float(), tensor(y_df.to_numpy()).float())


def train(loader, model, criterion, optimizer):
    model.train()
    for i, (X, y) in enumerate(loader):
        print(f"    {i + 1} / {len(loader)}", end="\r")
        X, y = X.to('cuda'), y.to('cuda')
        out = model(X)
        loss = criterion(out, y[:, None])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print()


def score(loader, model, criterion):
    with torch.no_grad():
        y_true, y_pred = predict(loader, model)
        return criterion(y_pred, y_true[:, None])


def predict(loader, model):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for X, y in loader:
            X, y = X.to('cuda'), y.to('cuda')
            y_pred = torch.cat((y_pred, model(X)), 0)
            y_true = torch.cat((y_true, y), 0)
    return y_true, y_pred


def evaluate(loader, model):
    true, pred = predict(loader, model)
    plt.scatter(true.cpu(), true.cpu(), s=3, label='True')
    plt.scatter(true.cpu(), pred.cpu(), s=3, label='Predicted')
    plt.legend()
    plt.show()
    plt.cla()


dataset = DataManager('../data/csv', rus=True)
X_train = dataset.X_train[dataset.y_train_binary.astype(bool)]
y_train = dataset.y_train[dataset.y_train_binary.astype(bool)]
train_dataset = get_tensordataset(X_train, y_train)
val_dataset = get_tensordataset(dataset.X_val, dataset.y_val)


model = nn.Sequential(
        nn.Linear(33, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
        nn.ReLU(),
).cuda()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
train_losses = []
val_losses = []


for epoch in range(1, 1000):
    train(train_loader, model, criterion, optimizer)
    with torch.no_grad():
        train_loss = score(train_loader, model, criterion)
        val_loss = score(val_loader, model, criterion)
        train_losses.append(train_loss.cpu())
        val_losses.append(val_loss.cpu())
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), '../model/ffnn_rus')

print("Train")
score(train_loader, model, criterion)
print("Val")
score(val_loader, model, criterion)
plt.plot(list(range(len(train_losses))), train_losses)
plt.plot(list(range(len(val_losses))), val_losses)
plt.show()
plt.cla()
evaluate(train_loader, model)
evaluate(val_loader, model)
