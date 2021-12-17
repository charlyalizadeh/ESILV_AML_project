from datamanager import DataManager
import torch
from torch import tensor
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, RMSprop
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.nn.functional import one_hot


def get_tensordataset(X_df, y_df):
    return TensorDataset(tensor(X_df.to_numpy()).float(), tensor(y_df.to_numpy()).float()[:, None])


def train(loader, model, criterion, optimizer):
    model.train()
    for i, (X, y) in enumerate(loader):
        print(f"    {i + 1} / {len(loader)}", end="\r")
        X, y = X.to('cuda'), y.to('cuda')
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print()


def score(loader, model, criterion):
    with torch.no_grad():
        y_true, y_pred = predict(loader, model)
        return criterion(y_pred, y_true)


def predict(loader, model):
    with torch.no_grad():
        model.eval()
        y_true = tensor([]).cuda()
        y_pred = tensor([]).cuda()
        for X, y in loader:
            X, y = X.to('cuda'), y.to('cuda')
            y_pred = torch.cat((y_pred, model(X)), 0)
            y_true = torch.cat((y_true, y), 0)
    return y_true, torch.round(y_pred)


def evaluate_classification(loader, model):
    true, pred = predict(loader, model)
    cm = confusion_matrix(true.cpu(), pred.cpu())
    accuracy = accuracy_score(true.cpu(), pred.cpu())
    print(f'Confusion matrix:\n{cm}')
    print(f'Accuracy: {accuracy}')


dataset = DataManager('../data/csv', rus=True)
X_train = dataset.X_train
y_train = dataset.y_train_binary
train_dataset = get_tensordataset(X_train, y_train)
val_dataset = get_tensordataset(dataset.X_val, dataset.y_val_binary)


model = nn.Sequential(
        nn.Linear(33, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 1),
        nn.Sigmoid()
).cuda()
optimizer = Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
train_losses = []
val_losses = []


for epoch in range(1, 100):
    train(train_loader, model, criterion, optimizer)
    with torch.no_grad():
        train_loss = score(train_loader, model, criterion)
        val_loss = score(val_loader, model, criterion)
        train_losses.append(train_loss.cpu())
        val_losses.append(val_loss.cpu())
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), '../model/ffnn_model_wo_null')

print("Train")
score(train_loader, model, criterion)
print("Val")
score(val_loader, model, criterion)
plt.plot(list(range(len(train_losses))), train_losses)
plt.plot(list(range(len(val_losses))), val_losses)
plt.show()
plt.cla()
evaluate_classification(train_loader, model)
evaluate_classification(val_loader, model)
