#%% Imports and Setup
from cProfile import label
from gzip import FNAME
from time import time
from turtle import speed
import keras
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from urllib.parse import quote
import os
import random
import pandas as pd
from io import BytesIO
import requests
import logging
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, Flatten
from tensorflow.keras import Sequential

#%% Logging Configuration
logging.basicConfig(level=logging.DEBUG, filename='error_log.txt', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

#%% Data Loading and Preprocessing
tf.compat.v1.enable_eager_execution()
df = pd.read_csv("/mnt/c/Users/ansh jha/OneDrive/Desktop/project/train.csv")
base_path = ("/mnt/c/Users/ansh jha/OneDrive/Desktop/project/images")
df = df.loc[df["id"].str.startswith(('00','b7','d1'),na=False),:]
data = pd.DataFrame(df["landmark_id"].value_counts())
num_classes = len(df["landmark_id"].value_counts())

# Data Visualization (commented out)
# plt.hist(data['count'],100,range=(16,32))
# plt.show()
print(df.columns)
print(df.shape)

#%% Label Encoding
labelenc = LabelEncoder()
labelenc.fit(df["landmark_id"])

def encode_label(label):
  if isinstance(label, (list, np.ndarray)):
    return labelenc.transform(label)
  else:
    return labelenc.transform([label])
  
def decode_label(label):
  return labelenc.inverse_transform([label])

#%% Image Fetching Function
def get_image_from_number(num, df, retries=3):
  fname, label = df.iloc[num]['url'], df.iloc[num]['landmark_id']
  encoded_fname = quote(fname, safe=':/')
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
  }
  for attempt in range(retries):
    try:
      response = requests.get(encoded_fname, timeout=300)
      if response.status_code == 200:
        img = np.array(Image.open(BytesIO(response.content)))
        return img, label
      else:
        print(f"Image not found or could not be fetched: {fname}")
        return None, None
    except requests.exceptions.Timeout:
      print(f"Timeout fetching image from {fname}. Retrying... ({attempt+1}/{retries})")
    except Exception as e:
      print(f"Error fetching image from {fname}: {e}")
      return None, None
  return None, None

#%% Model Architecture
learning_rate = 0.0001
decay_speed = 1e-6
momentum = 0.09
loss_function = "sparse_categorical_crossentropy"
source_mode = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Sequential()
for layer in source_mode.layers[:-1]:
  model.add(layer)
# Commenting out the conditional BatchNormalization as it's a bit ambiguous
# if layer == source_mode.layers[-22]:
#   model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
optiml = keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=optiml, loss=loss_function, metrics=["accuracy"])

#%% Utility Functions
def image_reshape(im, target_size):
  return cv2.resize(im, target_size)

def get_batch(data, start, batch_size):
  image_array = []
  label_array = []
  end_img = start + batch_size
  if end_img > len(data):
    end_img = len(data)
  for idx in range(start, end_img):
    im, label = get_image_from_number(idx, data)
    if im is None:
      print(f"Skipping image at index {idx} due to fetch failure.")
      continue
    im = image_reshape(im, (224, 224)) / 255.0
    image_array.append(im)
    label_array.append(label)
  label_array = encode_label(label_array)
  return np.array(image_array), np.array(label_array)

#%% Training and Validation Setup
batch_size = 16
epoch_shuffle = True
weight_classes = True
epochs = 1

train, val = np.split(df.sample(frac=1), [int(0.8 * len(df))])

#%% Training Loop
for e in range(epochs):
  print(f"Epoch: {e + 1}/{epochs}")
  train = train.sample(frac=1)
  for it in range(int(np.ceil(len(train) / batch_size))):
    X_train, Y_train = get_batch(train, it * batch_size, batch_size)
    if X_train.size > 0 and Y_train.size > 0:
      print(f"Training on batch {it + 1}/{int(np.ceil(len(train) / batch_size))}")
      model.train_on_batch(tf.convert_to_tensor(X_train), tf.convert_to_tensor(Y_train))
    else:
      print(f"Skipping batch {it+1} due to empty data.")
      
#%% Test Phase
batch_size = 16
errors = 0
good_preds = []
bad_preds = []

for it in range(int(np.ceil(len(val) / batch_size))):
  X_val, Y_val = get_batch(val, it * batch_size, batch_size)
  result = model.predict(X_val)
  cla = np.argmax(result, axis=1)

  for idx, res in enumerate(result):
    if cla[idx] != Y_val[idx]:
      errors += 1
      bad_preds.append([batch_size * it + idx, cla[idx], res[cla[idx]]])
    else:
      good_preds.append([batch_size * it + idx, cla[idx], res[cla[idx]]])

#%% Visualization
for i in range(min(5, len(good_preds))):
  n = int(good_preds[i][0])
  img, _ = get_image_from_number(n, val)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.imshow(img)
  plt.axis('off')

plt.show()