import json
import random

def split_data(intents, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    train_data = {"intents": []}
    val_data = {"intents": []}
    test_data = {"intents": []}
    
    for intent in intents["intents"]:
        patterns = intent["patterns"]
        random.shuffle(patterns)
        n_total = len(patterns)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_patterns = patterns[:n_train]
        val_patterns = patterns[n_train:n_train + n_val]
        test_patterns = patterns[n_train + n_val:]
        
        train_data["intents"].append({"tag": intent["tag"], "patterns": train_patterns})
        val_data["intents"].append({"tag": intent["tag"], "patterns": val_patterns})
        test_data["intents"].append({"tag": intent["tag"], "patterns": test_patterns})
    
    return train_data, val_data, test_data

with open('intents.json', 'r') as json_file:
    intents = json.load(json_file)

train_data, val_data, test_data = split_data(intents)

with open('train_data.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open('cross_validation.json', 'w') as val_file:
    json.dump(val_data, val_file, indent=4)

with open('testing.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)
