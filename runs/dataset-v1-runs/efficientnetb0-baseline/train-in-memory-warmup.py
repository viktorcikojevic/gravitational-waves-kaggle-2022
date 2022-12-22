#!/usr/bin/env python
# coding: utf-8




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
import pandas as pd
from gradient_accumulator.GAModelWrapper import GAModelWrapper











BATCH_SIZE = 4
ACCUM_STEPS = 16
WANDB_LOG = False


df = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/dataset-v1/data.csv")


# df["file"] = df["file"].str.replace("max_time_2_mean", "max_time_2_mean_TMP")

# take rows with depth smaller than 20
df = df[df["depth"] < 15]
df = df.sample(frac=1).reset_index(drop=True)

df = df[df["depth"] > 13]
df = df.sample(frac=1).reset_index(drop=True)


# print the number of rows with label == True
print("Number of rows with label == True: ", len(df[df["label"] == True]))
print("Number of rows with label == False: ", len(df[df["label"] == False]))



#  Take only 40,000 samples
# df = df.sample(n=40000, random_state=42)

# take only the rows with 'depth' lower than 11
# df = df[df['depth'] < 12]
# reset the index
df = df.reset_index(drop=True)

# remove 35% of the noise samples from the dataframe in this way: df = df[(df["label"] == 1) | (np.random.random(len(df)) < 0.65)] outside this script.
# df = df[(df["label"] == 1) | (np.random.random(len(df)) < 0.7)]

# Get all the unique names from the dataframe
names = df["file"].unique()
train_names, test_names = train_test_split(names, test_size=0.1, random_state=42)

# create df_train and df_test dataframes
df_train = df[df["file"].isin(train_names)]
df_test = df[df["file"].isin(test_names)]



# Get all the "file" column values into a list, where label is True
signal_data_train = df_train[df_train["label"] == True]["file"].tolist()
signal_data_test = df_test[df_test["label"] == True]["file"].tolist()
# Get all the "file" column values into a list, where label is False
noise_data_train = df_train[df_train["label"] == False]["file"].tolist()
noise_data_test = df_test[df_test["label"] == False]["file"].tolist()

train_files = signal_data_train + noise_data_train
test_files = signal_data_test + noise_data_test

# Create labels for train and test data
train_signal_labels = np.array([1 if "True" in file else 0 for file in train_files])
test_signal_labels = np.array([1 if "True" in file else 0 for file in test_files])

# train_detector_labels = np.array([1 if "H1" in file else 0 for file in train_files])
# test_detector_labels = np.array([1 if "H1" in file else 0 for file in test_files])

# train_depth_labels = df[df["file"].isin(train_files)]["depth"].values
# test_depth_labels = df[df["file"].isin(test_files)]["depth"].values

# # zip all the data into a list of tuples
# train_labels = list(zip(train_signal_labels, train_detector_labels, train_depth_labels))
# test_labels = list(zip(test_signal_labels, test_detector_labels, test_depth_labels))
train_labels = train_signal_labels
test_labels = test_signal_labels


# Load numpy arrays with tf.data.Dataset. Paths to numpy arrays are given in train_files and test_files. Data has a label 0 if it's a signal from h0, and 1 if it's a signal from h1.
train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
test_dataset = tf.data.Dataset.from_tensor_slices(test_files)




# Load numpy arrays from file paths. Load then in the following way:
# The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
def load_data(file_path, test_data=False):
    # Load the numpy arrays
    # The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
    file_path = file_path.numpy().decode("utf-8")
    
    # load the data
    data = np.load(file_path)
    
    label = 1 if "True" in file_path else 0
    return data, label
    
    


# call the load function on each element of the dataset using tf.py_function
train_dataset = train_dataset.map(lambda x: tf.py_function(load_data, [x], [tf.float32, tf.int32]))
# For test data, pass True to the load_data function
test_dataset = test_dataset.map(lambda x: tf.py_function(load_data, [x], [tf.float32, tf.int32]))

# # remove the last dimension of the data
train_dataset = train_dataset.map(lambda x, y: (tf.squeeze(x), y))
test_dataset = test_dataset.map(lambda x, y: (tf.squeeze(x), y))

# Shuffle and batch the data
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


# cache and prefetch the data
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(1)

test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(1)

# Repeat the data
train_dataset = train_dataset.repeat()
test_dataset = test_dataset.repeat()

# q: is the train dataset shuffled?
# a: yes, it is shuffled by default

# Make the dataset not shuffled
# train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=False)


# # take a look at the data
# for x, y in train_dataset.take(1):
#     print(x.shape)
#     print(y.shape)
#     print(x[0].shape)
#     print(y[0].shape)
    # print(x[0])
    # print(y[0])
    # print(x[0].numpy().shape)
    # print(y[0].numpy().shape)
    # print(x[0].numpy())
    # print(y[0].numpy())
# exit()







# load resnet model
base_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=True,
    weights=None,
    input_shape=(360, 256, 2),
    pooling=None,
    classes=1,
    classifier_activation='sigmoid',
)
inputs = tf.keras.Input(shape=(360, 256, 2))
x = base_model(inputs, training=True)
model = tf.keras.Model(inputs, x, name='EfficientNetB0')


# For each BatchNormalization layer in model.layers[-2].layers, set the momentum to 0.999
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        print("setting momentum to 0.999 for layer: ", layer.name)
        layer.momentum = 0.9


model = GAModelWrapper(accum_steps=ACCUM_STEPS, inputs=model.input, outputs=model.output)



# compile model. The model has 3 outputs, so we need to specify a loss for each output
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.E-09),
                # loss is binary crosseentropy, because we have a binary classification problem
                loss= tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
                metrics=[tf.keras.metrics.AUC()]
                )






# path = "init_model.h5"
# model.load_weights(path)


model.evaluate(test_dataset.take(64))

# exit()




# Train the model, and evaluate it on the test dataset. steps_per_epoch=4, epochs=10, validation_data=test_dataset, validation steps=4. Save the best model during training to 'best_model.h5'
date = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

# Set a learning rate callback, so that it is equal to 0.01 during first 10 epochs, and then it is equal to 0.1 afterwards
#lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.01 if epoch < 10 else 0.1)


if WANDB_LOG:
    import wandb
    from wandb.keras import WandbCallback

    wandb.init(project="gravitational-waves", entity="viktor-cikojevic")


# fit using the CPU
# with tf.device('/CPU:0'):
history = model.fit(train_dataset, 
        steps_per_epoch=128, 
        epochs=4, 
        validation_data=test_dataset, 
        validation_steps=64,
        # don't shuffle the data
        # shuffle=True,
        callbacks=[
            # WandbCallback(save_model=False), 
            # tf.keras.callbacks.ModelCheckpoint(f"best_model.h5", save_best_only=True, monitor='val_auc', mode='max')
            ]
        )




# perform the fit above on the CPU, not GPU
# 

# model.evaluate(test_dataset.take(256))


# Save the model
# model.save("signals_true_mix_vs_noise/last_model.h5")
model.save("warmup-model/model.h5")


# save history to a file in the same directory as the model
with open(f"history.txt", "w") as f:
    f.write(str(history.history))






