from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes=51

train_data=pd.read_csv('new_gait_dataset/only_sample_gait_dataset.csv')
test_data=pd.read_csv('new_gait_dataset/original_test_gait_dataset.csv')


x_train=train_data.iloc[:,:17]
y_train=train_data.iloc[:,17]

x_test=test_data.iloc[:,:17]
y_test=test_data.iloc[:,17]
Stand_X = StandardScaler() 
x_train = Stand_X.fit_transform(x_train)
x_test= Stand_X.fit_transform(x_test)




def knn():
    knn = KNeighborsClassifier(n_neighbors=5,weights='distance', metric='manhattan')

    knn.fit(x_train, y_train)  

    y_pred = knn.predict(x_test)

    print(knn.predict(x_test))  
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(knn, 'model/only_knn.pkl')

def svm():

    clf = SVC( C=15, kernel='rbf', decision_function_shape='ovr',probability=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(accuracy)
    joblib.dump(clf, 'model/only_svm.pkl')

def rf():
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    joblib.dump(forest, 'model/only_rf.pkl')






class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=17, out_channels=51, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)



def cnn():
    print("\nTraining CNN (PyTorch)...")
    model = CNN(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS_CNN):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS_CNN} - Loss: {total_loss/len(train_loader):.4f}")
     model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    print("CNN accuracy:", acc)
    print(classification_report(all_true, all_preds))
    torch.save(model.state_dict(), "model/cnn_torch.pth"))
    return model




class LSTM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=17, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_lstm_torch():
    print("\nTraining LSTM (PyTorch)...")
    model = LSTM(n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_seq, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(x_test_seq, dtype=torch.float32),
                      torch.tensor(y_test, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS_LSTM):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS_LSTM} - Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            preds = model(xb)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    print("LSTM accuracy:", acc)
    print(classification_report(all_true, all_preds))
    torch.save(model.state_dict(), "model/lstm_torch.pth"))
    return model





if __name__ == '__main__':
    svm()
    knn()
    rf()
    cnn()
    lstm()



