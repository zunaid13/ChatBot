import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize, stem

# Load model and other necessary data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the cross-validation data
with open('cross_validation.json', 'r') as json_file:
    cross_validation_data = json.load(json_file)

# Load the train data
with open('train_data.json', 'r') as json_file:
    train_data = json.load(json_file)

# Prepare data
all_words = []
tags = []
xy = []
for intent in train_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_data = []
y_data = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_data.append(bag)
    label = tags.index(tag)
    y_data.append(label)

X_data = np.array(X_data)
y_data = np.array(y_data)

# Cross-validation function
def cross_validate(model, X_data, y_data, k=10, num_epochs=1000, batch_size=8, learning_rate=0.001, hidden_size=8):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    train_errors = []
    test_errors = []

    for train_index, test_index in kf.split(X_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # Create dataset and dataloader
        train_dataset = ChatDataset(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        model = NeuralNet(input_size=len(X_train[0]), hidden_size=hidden_size, num_classes=len(tags)).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(num_epochs):
            for words, labels in train_loader:
                words = words.to(device)
                labels = labels.to(dtype=torch.long).to(device)

                outputs = model(words)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate the model on train and test data
        with torch.no_grad():
            train_preds = model(torch.from_numpy(X_train).to(device)).argmax(dim=1).cpu().numpy()
            test_preds = model(torch.from_numpy(X_test).to(device)).argmax(dim=1).cpu().numpy()
            train_accuracy = accuracy_score(y_train, train_preds)
            test_accuracy = accuracy_score(y_test, test_preds)
            train_errors.append(1 - train_accuracy)
            test_errors.append(1 - test_accuracy)

    return np.mean(train_errors), np.mean(test_errors)

class ChatDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.n_samples = len(X_data)
        self.x_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Define hyperparameters to test
hyperparameters = {
    "num_epochs": [500, 1000, 1500],
    "batch_size": [8, 16, 32],
    "learning_rate": [0.001, 0.01, 0.1],
    "hidden_size": [8, 16, 32]
}

# Perform cross-validation for each set of hyperparameters
results = []
for num_epochs in hyperparameters["num_epochs"]:
    for batch_size in hyperparameters["batch_size"]:
        for learning_rate in hyperparameters["learning_rate"]:
            for hidden_size in hyperparameters["hidden_size"]:
                print(f"Testing hyperparameters: num_epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}, hidden_size={hidden_size}")
                train_error, test_error = cross_validate(NeuralNet(len(X_data[0]), hidden_size, len(tags)), X_data, y_data, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, hidden_size=hidden_size)
                results.append((num_epochs, batch_size, learning_rate, hidden_size, train_error, test_error))

# Plot the results
epochs, batches, lrs, h_sizes, train_errors, test_errors = zip(*results)

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(train_errors, test_errors, c=h_sizes, cmap='viridis', s=100)
legend1 = ax.legend(*scatter.legend_elements(), title="Hidden Sizes")
ax.add_artist(legend1)
plt.xlabel("Train Error")
plt.ylabel("Test Error")
plt.title("Train vs Test Error for Different Hyperparameters")
plt.grid(True)
plt.show()
