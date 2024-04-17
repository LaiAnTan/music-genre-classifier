import tensorflow as tf
keras = tf.keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, TextVectorization, Normalization
from keras.callbacks import EarlyStopping
from keras import regularizers

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import scipy as sp
import platform
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Loading dataset 
data = pd.read_csv('../data/raw.csv')
data = data.dropna()

# Scaling and normalizing the data
df = data.copy()

# vectorise text data into int
artist_name_vectorizer = TextVectorization(output_mode='int')
artist_name_vectorizer.adapt(df['artist_name'])
artist_name_vectorized = artist_name_vectorizer(df['artist_name'])

# flatten
artist_name_vectorized = tf.reduce_mean(artist_name_vectorized, axis=-1)

df['artist_name'] = artist_name_vectorized.numpy()

track_name_vectorizer = TextVectorization(output_mode='int')
track_name_vectorizer.adapt(df['track_name'])
track_name_vectorized = track_name_vectorizer(df['track_name'])
0
# flatten
track_name_vectorized = tf.reduce_mean(track_name_vectorized, axis=-1)

df['track_name'] = track_name_vectorized.numpy()

# drop unrelated
df.drop(columns=['music_genre', 'instance_id'])

# Normalize
sclr = StandardScaler()

df = pd.DataFrame(sclr.fit_transform(df), columns=df.columns)

# Seperating features and the label
features = df.select_dtypes(include=[np.number])
features = features.drop(columns=['music_genre', 'instance_id'])
label = data.iloc[:, -1]

features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Building the model
model = Sequential()

model.add(Dense(13, input_dim=features_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(32, activation='relu', kernel_regularizer= regularizers.l2(0.01)))

model.add(Dense(32, activation='relu', kernel_regularizer= regularizers.l2(0.01)))

model.add(Dense(32, activation='relu', kernel_regularizer= regularizers.l2(0.01)))
# Output layer
model.add(Dense(10, activation = 'softmax'))

optimizer = keras.optimizers.Adam()

# Compiling the model

loss_fn = keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

# Define the early stopping criteria
stop_early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Training the model
model.fit(features_train, label_train, epochs=200, batch_size=32, validation_data=(features_test, label_test), callbacks=[stop_early])

# Evaluate the model
loss, accuracy = model.evaluate(features_test, label_test)
print(f'Accuracy: {accuracy*100}%')

model.save("model.keras")