from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from datamanager import DataManager
from joblib import dump

dataset = DataManager('../data/csv')

clf = RandomForestClassifier(random_state=42)
clf.fit(dataset.X_train, dataset.y_train_binary)
dump(clf, '../model/random_forest.joblib')

y_train_pred = clf.predict(dataset.X_train)
y_val_pred = clf.predict(dataset.X_val)

print(f'Train accuracy: {accuracy_score(dataset.y_train_binary, y_train_pred)}')
print(f'Train f1: {f1_score(dataset.y_train_binary, y_train_pred)}')
print(f'Train confusion matrix:\n{confusion_matrix(dataset.y_train_binary, clf.predict(dataset.X_train))}')
print(f'Val accuracy: {accuracy_score(dataset.y_val_binary, y_val_pred)}')
print(f'Val f1: {f1_score(dataset.y_val_binary, y_val_pred)}')
print(f'Val confusion matrix:\n{confusion_matrix(dataset.y_val_binary, clf.predict(dataset.X_val))}')
