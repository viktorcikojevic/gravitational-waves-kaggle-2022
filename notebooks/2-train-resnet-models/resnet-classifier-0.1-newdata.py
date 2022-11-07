#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
# Import plt from matplotlib
import matplotlib.pyplot as plt
import glob
import random
# Import train_test_split from sklearn
from sklearn.model_selection import train_test_split
import json
import os
import datetime
import time
import shutil


# In[2]:


import wandb
from wandb.keras import WandbCallback

wandb.init(project="gravitational-waves", entity="viktor-cikojevic")


# In[3]:


BATCH_SIZE = 8
STEPS_PER_EPOCH = 16
VALIDATION_STEPS = 4
EPOCHS = 100


# In[4]:


# Read the data.
# root = "/Volumes/T7/gravitational-waves"
root = "/media/viktor/T7/gravitational-waves"

all_files =  glob.glob(f'{root}/generated-signals/signals_mix/*.npy')[:31000] + glob.glob(f'{root}/generated-signals/signals_noise/*.npy')
print("There are {} files".format(len(all_files)))
# Shuffle all_files
random.shuffle(all_files)
# Train/test split all_files
train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)

# Create labels for train and test data
train_labels = np.array([int(file.split("/")[-2] == 'signals_mix') for file in train_files])
test_labels = np.array([int(file.split("/")[-2] == 'signals_mix') for file in test_files])

# Load numpy arrays with tf.data.Dataset. Paths to numpy arrays are given in train_files and test_files. Data has a label 0 if it's a signal from h0, and 1 if it's a signal from h1.
train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
# Load numpy arrays from file paths
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

train_dataset = train_dataset.map(lambda x: tf.numpy_function(np.load, [x], [tf.float32]))
test_dataset = test_dataset.map(lambda x: tf.numpy_function(np.load, [x], [tf.float32]))

# restore np.load for future normal usage
np.load = np_load_old

# _pickle.UnpicklingError: Failed to interpret file b'/media/viktor/T7/gravitational-waves/generated-signals/signals_mix/round_2630428_0_5.npy' as a pickle


# Create tf.data.Dataset from train_labels and test_labels
train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
test_labels = tf.data.Dataset.from_tensor_slices(test_labels)

# Combine train_dataset and train_labels
train_dataset = tf.data.Dataset.zip((train_dataset, train_labels))
test_dataset = tf.data.Dataset.zip((test_dataset, test_labels))

# Shuffle and batch the data
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)




# Repeat the data
train_dataset = train_dataset.repeat()
test_dataset = test_dataset.repeat()

tmp_train = next(train_dataset.as_numpy_iterator())
tmp_test = next(test_dataset.as_numpy_iterator())
# tmp_train[0][0].shape
tmp_train[0][0].shape, tmp_train[1]


# In[5]:


np.sum(["signals_noise" in file  for file in test_files]) / len(test_files), len(test_files), len(train_files)


# In[6]:


# Create a ResNet model. Load from keras.applications.resnet50.ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
# Create a ResNet50 model with weights not pre-trained on  ImageNet. Set include_top to False.  Set input_shape to (360, 360, 4). Set pooling to 'avg'. Classify into 2 classes. Activate sigmoid function.
resnet = ResNet50(weights=None, include_top=False, input_shape=(360, 360, 4), pooling='avg', classes=1)
model = tf.keras.Sequential([
    tf.keras.layers.Permute((2, 3, 1), input_shape=(4, 360, 360)),
    resnet,
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])


model.compile(optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.02), 
              loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
              metrics=[tf.keras.metrics.AUC()]
              )
model.summary()





# In[7]:


# path = "resnet-classifier/signals_true_mix_vs_noise/29-10-2022_22:23:46/best_model.h5"
# model.load_weights(path)


# In[8]:


# tf.keras.utils.plot_model(model, show_shapes=True)


# In[9]:


model.evaluate(train_dataset.take(4))


# In[10]:


# Evaluate the model on the test dataset. 
# model(tmp_test[1]).shape
model.evaluate(test_dataset.take(4))


# In[11]:


# Train the model, and evaluate it on the test dataset. steps_per_epoch=4, epochs=10, validation_data=test_dataset, validation steps=4. Save the best model during training to 'best_model.h5'
date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

# Set a learning rate callback, so that it is equal to 0.01 during first 10 epochs, and then it is equal to 0.1 afterwards
#lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 if epoch < 10 else 0.1)




history = model.fit(train_dataset, 
          steps_per_epoch=500, 
          epochs=1000, 
          validation_data=test_dataset, 
          validation_steps=100,
          callbacks=[WandbCallback()]
          )

# Save the model
model.save("resnet-classifier/signals_true_mix_vs_noise/{}/best_model.h5".format(date))

# In[ ]:




