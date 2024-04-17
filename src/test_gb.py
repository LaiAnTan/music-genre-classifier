from train_gb import features_test, label_test
from sklearn.metrics import accuracy_score

import pickle

# Load the model from disk
filename = '../models/xgbclassifier_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# Now you can use `loaded_model` to make predictions
# For example, to predict the first 10 instances of your test set:
predictions = loaded_model.predict(features_test)

print(predictions)

# Check accuracy
accuracy = accuracy_score(label_test, predictions)

print(f"Accuracy: {accuracy * 100.0}%")