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
import argparse

import wandb
from wandb.keras import WandbCallback





def train(depth_min):



    BATCH_SIZE = 4
    ACCUM_STEPS = 16
    WANDB_LOG = False


    df = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/dataset-v1/data.csv")


    # df["file"] = df["file"].str.replace("max_time_2_mean", "max_time_2_mean_TMP")

    # take rows with depth smaller than 20
    df = df[df["depth"] < depth_min+2].reset_index(drop=True)
    df = df[df["depth"] > depth_min].reset_index(drop=True)
    print(df)


    # print the number of rows with label == True
    print("Number of rows with label == True: ", len(df[df["label"] == 1]))
    print("Number of rows with label == False: ", len(df[df["label"] == 0]))

    # Get all the unique names from the dataframe
    names = df["file"].unique()
    train_names, test_names = train_test_split(names, test_size=0.1, random_state=42)

    # create df_train and df_test dataframes
    df_train = df[df["file"].isin(train_names)]
    df_test = df[df["file"].isin(test_names)]



    # Get all the "file" column values into a list, where label is True
    signal_data_train = df_train[df_train["label"] == 1]["file"].tolist()
    signal_data_test = df_test[df_test["label"] == 1]["file"].tolist()
    # Get all the "file" column values into a list, where label is False
    noise_data_train = df_train[df_train["label"] == 0]["file"].tolist()
    noise_data_test = df_test[df_test["label"] == 0]["file"].tolist()

    train_files = signal_data_train + noise_data_train
    test_files = signal_data_test + noise_data_test

    # Create labels for train and test data
    train_labels = df_train[df_train["label"] == 1]["label"].tolist() + df_train[df_train["label"] == 0]["label"].tolist()
    test_labels = df_test[df_test["label"] == 1]["label"].tolist() + df_test[df_test["label"] == 0]["label"].tolist()

    # shuffle train_files and train_labels in the same way
    import random
    random.seed(42)
    random.shuffle(train_files)
    random.shuffle(test_files)
    

    # Load numpy arrays with tf.data.Dataset. Paths to numpy arrays are given in train_files and test_files. Data has a label 0 if it's a signal from h0, and 1 if it's a signal from h1.
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
    
    

    


    # Load numpy arrays from file paths. Load then in the following way:
    # The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
    def load_data(file_path, test_data=False):
        # Load the numpy arrays
        # The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
        file_path = file_path.numpy().decode("utf-8")
        
        
        # print(file_path)
        
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
    
    # shuffle the data
    train_dataset = train_dataset.shuffle(buffer_size=16)
    test_dataset = test_dataset.shuffle(buffer_size=16)

    # Shuffle and batch the data
    train_dataset = train_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    

    # Repeat the data
    train_dataset = train_dataset.repeat()
    test_dataset = test_dataset.repeat()

    # cache and prefetch the data
    train_dataset = train_dataset.prefetch(1)
    test_dataset = test_dataset.prefetch(1)
    
    # q: is the train dataset shuffled?
    # a: yes, it is shuffled by default

    # Make the dataset not shuffled
    # train_dataset = train_dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=False)


    # take a look at the data
    for x, y in train_dataset.take(1):
        
        
        print(y)
        # print(x[0].shape)
        # print(y[0].shape)
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
    model = tf.keras.Model(inputs, x)


    # --- Warmup ---

    # For each BatchNormalization layer in model.layers[-2].layers, set the momentum to 0.999
    for layer in model.layers[-1].layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            init_momentum = 0.9
            print(f"setting momentum to {init_momentum} for layer: ", layer.name)
            layer.momentum = init_momentum

    
    for layer in model.layers[-1].layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            print(f"double check", layer.name, layer.momentum)
    
    
    model = GAModelWrapper(accum_steps=ACCUM_STEPS, inputs=model.input, outputs=model.output)



    # compile model. The model has 3 outputs, so we need to specify a loss for each output
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.E-09),
                    # loss is binary crosseentropy, because we have a binary classification problem
                    loss= tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
                    metrics=[tf.keras.metrics.AUC()]
                    )

    history = model.fit(train_dataset, 
            steps_per_epoch=128, 
            epochs=1, 
            validation_data=test_dataset, 
            validation_steps=64,
            # don't shuffle the data
            shuffle=True,
            callbacks=[
                # WandbCallback(save_model=False), 
                # tf.keras.callbacks.ModelCheckpoint(f"best_model/best_model.h5", save_best_only=True, monitor='val_auc', mode='max')
                ]
            )

    # --- Fine-tune ---
    model.optimizer.learning_rate = 1.E-04

    for layer in model.layers[-1].layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            init_momentum = 0.99
            print(f"setting momentum to {init_momentum} for layer: ", layer.name)
            layer.momentum = init_momentum
    
    
    try:
        wandb.init(project="gravitational-waves", entity="viktor-cikojevic", name=f"depth_min_{depth_min}")
    except:
        wandb.init(project="gravitational-waves", entity="viktor-cikojevic")
    
    
    
    history = model.fit(train_dataset, 
            steps_per_epoch=128, 
            epochs=256, 
            validation_data=test_dataset, 
            validation_steps=64,
            # don't shuffle the data
            shuffle=True,
            callbacks=[
                WandbCallback(save_model=False), 
                tf.keras.callbacks.ModelCheckpoint(f"best_model/best_model_{depth_min}.h5", save_best_only=True, monitor='val_auc', mode='max')
                ]
            )




    # perform the fit above on the CPU, not GPU
    # 

    # model.evaluate(test_dataset.take(256))


    # Save the model
    # model.save("signals_true_mix_vs_noise/last_model.h5")
    model.save("last_model/model_{depth_min}.h5")


    # save history to a file in the same directory as the model
    with open(f"histories/history_{depth_min}.txt", "w") as f:
        f.write(str(history.history))






# Main wil read depth_min, depth_max from the command line. It will then call the train function with those values
if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser(description='Train a model to detect signals in noise')
    parser.add_argument('depth_min', type=int, help='the minimum depth of the signal')
    args = parser.parse_args()

    train(args.depth_min)
