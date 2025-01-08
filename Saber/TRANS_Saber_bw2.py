import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

data = pd.read_csv('Saber_bw2_PoI_100000.csv', header=None)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

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

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_sampled, test_size=0.2, random_state=42)

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

batch_size = 256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, n_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=n_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim
        )

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        x = x.unsqueeze(0)

        transformer_out = self.transformer(x, x)

        output = transformer_out[0]

        output = self.fc_out(output)

        return output


input_dim = X_train.shape[1]
num_classes = 46
model = TransformerModel(input_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

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