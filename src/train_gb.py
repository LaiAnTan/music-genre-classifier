import xgboost as xgb
import pickle

import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd

data = pd.read_csv('../data/raw.csv')
data = data.dropna()

# Vectorize 'artist_name' and 'track_name' using TF-IDF
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5)
features_artist = pd.DataFrame(tfidf.fit_transform(data['artist_name']).toarray(), columns=[f'artist_{i}' for i in range(tfidf.fit_transform(data['artist_name']).shape[1])])
features_track = pd.DataFrame(tfidf.fit_transform(data['track_name']).toarray(), columns=[f'track_{i}' for i in range(tfidf.fit_transform(data['track_name']).shape[1])])

label = data['music_genre']

data = data.drop(columns=['artist_name', 'track_name', 'music_genre'])

features = pd.concat([data, features_artist, features_track], axis=1)

le = LabelEncoder()
label = le.fit_transform(label)

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=42)

model = XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), verbosity=2, max_depth=5)
model.fit(features_train, label_train)

label_pred = model.predict(features_test)

accuracy = accuracy_score(label_test, label_pred)
print(f"Accuracy: {accuracy * 100.0}%")

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))