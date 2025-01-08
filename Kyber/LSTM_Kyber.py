import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

file_path = 'Kyber_PoI_1000000.csv'
data = pd.read_csv(file_path, header=None)

X = data.iloc[:, :-1].to_numpy()
y = data.iloc[:, -1].to_numpy()

data['label'] = y
sampled_data = pd.DataFrame()

for label in np.unique(y):
    label_data = data[data['label'] == label]
    sampled_label_data = label_data.sample(n=100, random_state=42)
    sampled_data = pd.concat([sampled_data, sampled_label_data])

X_sampled = sampled_data.iloc[:, :-2].to_numpy()
y_sampled = sampled_data.iloc[:, -1].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_classes, num_layers, dropout_rate=0.5):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        lstm_out_last = lstm_out[:, -1, :]

        x = self.fc(lstm_out_last)

        return x


input_size = 40
hidden_dim = 64
num_classes = len(set(y))
num_layers = 2
dropout_rate = 0.5

model = LSTMModel(input_size=input_size, hidden_dim=hidden_dim, num_classes=num_classes,
                  num_layers=num_layers, dropout_rate=dropout_rate).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
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
