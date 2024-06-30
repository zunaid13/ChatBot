import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load model and other necessary data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the testing data
try:
    with open('testing.json', 'r') as json_file:
        testing_data = json.load(json_file)
    print("Loaded testing data.")
except Exception as e:
    print(f"Error loading testing.json: {e}")
    exit()

# Load the model state
try:
    FILE = "data.pth"
    data = torch.load(FILE)
    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()
    print("Loaded model and state data.")
except Exception as e:
    print(f"Error loading model data from {FILE}: {e}")
    exit()

y_true = []
y_pred = []

# Perform predictions
try:
    for intent in testing_data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            sentence = tokenize(pattern)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            y_true.append(tag)
            y_pred.append(tags[predicted.item()])

    print("Finished predictions.")
except Exception as e:
    print(f"Error during predictions: {e}")
    exit()

# Generate the confusion matrix
try:
    cm = confusion_matrix(y_true, y_pred, labels=tags)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Normalize and convert to percentage
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=tags)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', values_format=".2f")
    plt.xticks(rotation=90)
    plt.savefig('confusion_matrix_percentage.png')  # Save the plot to a file
    plt.show()
    print("Confusion matrix plotted and saved as confusion_matrix_percentage.png.")
except Exception as e:
    print(f"Error generating or plotting confusion matrix: {e}")
    exit()

# Print classification report
try:
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=tags))
except Exception as e:
    print(f"Error generating classification report: {e}")
