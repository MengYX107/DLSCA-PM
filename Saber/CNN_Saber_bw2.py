import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

file_path = 'Saber_bw2_PoI_100000.csv'
data = pd.read_csv(file_path, header=None)

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()
print(len(set(y)))
bb = {}
cc = 0
for aa in set(y):
    bb[aa] = cc
    cc += 1

for i in range(0,len(y)):
    y[i] = bb[y[i]]

data['label'] = y
sampled_data = pd.DataFrame()

for label in np.unique(y):
    label_data = data[data['label'] == label]
    sampled_label_data = label_data.sample(n=1000, random_state=42)
    sampled_data = pd.concat([sampled_data, sampled_label_data])

X_sampled = sampled_data.iloc[:, :-2].to_numpy()
y_sampled = sampled_data.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

batch_size = 256

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(dropout_rate)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * (input_size // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        x = self.pool(torch.relu(self.conv2(x)))

        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)

        return x


input_size = X_train.shape[1]
output_size = len(set(y))
model = CNNModel(input_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

num_epochs = 1000
patience = 20
min_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    if avg_loss < min_loss:
        min_loss = avg_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    if epochs_without_improvement >= patience:
        print(f"Stopping early due to no improvement in loss for {patience} epochs.")
        break

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')
print(f'Recall of the model on the test set: {recall * 100:.2f}%')
print(f'F1 Score of the model on the test set: {f1 * 100:.2f}%')
