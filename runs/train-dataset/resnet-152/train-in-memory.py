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

from PIL import Image

from sklearn.utils import shuffle


def train(depth_min):



    BATCH_SIZE = 2
    ACCUM_STEPS = 20
    WANDB_LOG = False


    # df = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/dataset-v1/data.csv")
    
    
    df = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/kaggle-data/preprocessed-2/train-preprocessed.csv")
    df["preprocessed_filename"] = df["preprocessed_filename"].apply(lambda x: x.replace("/media/viktor/T7/gravitational-waves-kaggle-2022", "/media/viktor/T7/gravitational-waves-kaggle-2022/kaggle-data/preprocessed-2"))
    # rename "preprocessed_filename" to "file" and "target" to "label"
    df.rename(columns={"preprocessed_filename": "file", "target": "label"}, inplace=True)
    df = df[df["label"] != -1].reset_index(drop=True)
    
    
    # df_2 = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/dataset-v2/data.csv")
    # # concatenate the two dataframes
    # df = pd.concat([df, df_2], ignore_index=True)
    
    # Get /Users/viktorcikojevic/Documents/gravitational-waves-kaggle-2022/datasets/realistic_noise_256/realistic_noise_256.csv as df_noise
    df_noise_h1 = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/realistic_noise_256/realistic_noise_256_h1.csv")
    df_noise_l1 = pd.read_csv("/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/realistic_noise_256/realistic_noise_256_l1.csv")
    folders_h1 = list(set(df_noise_h1["folder"].tolist()))
    folders_l1 = list(set(df_noise_l1["folder"].tolist()))
    

    # print the number of rows with label == True
    print("Number of rows with label == True: ", len(df[df["label"] == 1]))
    print("Number of rows with label == False: ", len(df[df["label"] == 0]))

    # Get all the unique names from the dataframe
    names = df["file"].unique()
    train_names, test_names = train_test_split(names, test_size=0.15, random_state=42)

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

    train_labels = [1] * len(signal_data_train) + [0] * len(noise_data_train)
    test_labels = [1] * len(signal_data_test) + [0] * len(noise_data_test)

    
    # shuffle train_files and train_labels in the same way
    train_files, train_labels = shuffle(train_files, train_labels)
    # shuffle test_files and test_labels in the same way
    test_files, test_labels = shuffle(test_files, test_labels)
    

    
    
    # Load numpy arrays with tf.data.Dataset. Paths to numpy arrays are given in train_files and test_files. Data has a label 0 if it's a signal from h0, and 1 if it's a signal from h1.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    
    

    


    # Load numpy arrays from file paths. Load then in the following way:
    # The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
    def load_data(file_path, label, test_data=False):
        # Load the numpy arrays
        # The file names are in the following format: 'amplitudes_a80309059_0_17.739233746429086_H1_True.npy' and 'amplitudes_a80309059_0_17.739233746429086_L1_True.npy'. Load both files and put them in each channel. The label is the same for both files. The label is 1 if the file name contains 'True', and 0 if it contains 'False'.
        file_path = file_path.numpy().decode("utf-8")
        
        
        # print(file_path)
        
        # load the data
        data = np.load(file_path)
        
        data *= 1. + 0.05 * np.random.randn(*data.shape)
    
        # randomly shift the data
        # shift = np.random.randint(-100, 200)
        # data = np.roll(data, shift, axis=0)
        # shift = np.random.randint(-100, 200)
        # data = np.roll(data, shift, axis=1)
        
        
        # randomly flip the data AROUND THE Y AXIS
        if np.random.rand() > 0.25:
            data = np.flip(data, axis=0)
        # randomly flip the data AROUND THE X AXIS
        if np.random.rand() > 0.25:
            data = np.flip(data, axis=1)
        

        # clip data to [0, 255]
        data = np.clip(data, 0, 255)
        
        
        
        # # Data augmentation with df_noise
        # # Get the random folder from folders
        
        # if np.random.rand() > 0.5:
        
        #     random_folder_h1 = random.choice(folders_h1)
        #     random_folder_l1 = random.choice(folders_l1)        
            
        #     random_H1 = np.array(Image.open(f"{random_folder_h1}/H1.png"))
        #     random_L1 = np.array(Image.open(f"{random_folder_l1}/L1.png"))
            
            
            
            
        #     random_data = np.stack([random_H1, random_L1], axis=2)
            
        #     # mix the data with the random data as x*data + (1-x)*random_data, where x is a random number between 0.3 and 1
        #     x = np.random.uniform(0.5, 1.)
        #     data = random_data + x * data
        #     data = 255 * data / np.max(data)
            
        #     # clip data to [0, 255]
        #     data = np.clip(data, 0, 255)
            
        
                
        
        data = 2 * data / 255.
        data -= 1.
        
        return data, label
        
        


    # call the load function on each element of the dataset using tf.py_function
    train_dataset = train_dataset.map(lambda x, y: tf.py_function(load_data, [x, y], [tf.float32, tf.int32]))
    test_dataset = test_dataset.map(lambda x, y: tf.py_function(load_data, [x, y], [tf.float32, tf.int32]))
    

    # # remove the last dimension of the data
    train_dataset = train_dataset.map(lambda x, y: (tf.squeeze(x), y))
    test_dataset = test_dataset.map(lambda x, y: (tf.squeeze(x), y))
    
    # shuffle the data
    # train_dataset = train_dataset.shuffle(buffer_size=16)
    # test_dataset = test_dataset.shuffle(buffer_size=16)

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







    def largeKernelInitializer():
        inputs = tf.keras.layers.Input(shape=(360, 256, 2))
        x = tf.keras.layers.Conv2D(2, (31, 31), strides=(2, 2), padding="same", kernel_initializer="he_normal")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(2, (31, 31), strides=(1, 1), padding="same", kernel_initializer="he_normal")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x, name="largeKernelInitializer")


    def create_model():
        base_model = tf.keras.applications.ResNet152V2(include_top=True, 
                                weights=None,
                                input_tensor=tf.keras.Input(shape=(360, 256, 2)),
                                classifier_activation="sigmoid",
                                classes=1
                                    )
        return base_model
        
        
        inputs = tf.keras.layers.Input(shape=(360, 256, 2))
        # x = largeKernelInitializer()(inputs)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x)
        return model



    model = create_model()
    model.summary()
    # exit()
    
    model = GAModelWrapper(accum_steps=ACCUM_STEPS, inputs=model.input, outputs=model.output)
    
    
    # Load weights from best_models/best_model_10.h5
    # model.load_weights("input-model/model.h5")
    
    
    
    


    # --- Warmup ---

    # For each BatchNormalization layer in model.layers[-2].layers, set the momentum to 0.999
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            init_momentum = 0.95
            print(f"setting momentum to {init_momentum} for layer: ", layer.name)
            layer.momentum = init_momentum

    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            print(f"double check", layer.name, layer.momentum)
    


    print("[INFO] compiling model...")
    # compile model. The model has 3 outputs, so we need to specify a loss for each output
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5.E-06),
                    # loss is binary crosseentropy, because we have a binary classification problem
                    loss= tf.keras.losses.BinaryFocalCrossentropy(gamma=4),
                    # loss= 'mse', # tf.keras.losses.BinaryFocalCrossentropy(gamma=4), # 0.68
                    metrics=[tf.keras.metrics.AUC()]
                    )
    
    
    history = model.fit(train_dataset, 
            steps_per_epoch=128, 
            epochs=1, 
            validation_data=test_dataset, 
            validation_steps=64,
            # don't shuffle the data
            shuffle=True,
            # callbacks=[
            #     WandbCallback(save_model=False), 
            #     tf.keras.callbacks.ModelCheckpoint(f"best_model/best_model.h5", save_best_only=True, monitor='val_auc', mode='max')
            #     ]
            )
    
    
    # set the learning rate to 5.E-5
    model.optimizer.learning_rate = 1.E-4
    
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            init_momentum = 0.9999
            print(f"setting momentum to {init_momentum} for layer: ", layer.name)
            layer.momentum = init_momentum

    
    
    print("[INFO] evaluating model...")
    model.evaluate(test_dataset, steps=128)
    # exit()
    
    try:
        wandb.init(project="gravitational-waves", entity="viktor-cikojevic", name=f"v1+v2+augm")
    except:
        wandb.init(project="gravitational-waves", entity="viktor-cikojevic")
    
    
    
    # Make a custom callback that saves the model as model.save("best_model.h5") after each epoch
    class SaveBest(tf.keras.callbacks.Callback):
        
        
        def __init__(self):
            super(SaveBest, self).__init__()
            self.best = 0
        
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_auc') > self.best:
                self.best = logs.get('val_auc')
                print(f"\nSaving model with val_auc: {self.best}")
                model.save("best_model/best_model.h5")
    
    
    history = model.fit(train_dataset, 
            steps_per_epoch=128, 
            epochs=1024, 
            validation_data=test_dataset, 
            validation_steps=64,
            # don't shuffle the data
            shuffle=True,
            callbacks=[
                WandbCallback(save_model=False), 
                SaveBest()
                ]
            )




    # perform the fit above on the CPU, not GPU
    # 

    # model.evaluate(test_dataset.take(256))


    # Save the model
    # model.save("signals_true_mix_vs_noise/last_model.h5")
    model.save("last_model/model.h5")


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
