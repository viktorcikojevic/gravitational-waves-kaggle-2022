#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import tensorflow as tf
import glob 
import pandas as pd
import h5py # import to read hdf5
from pathlib import Path
from tqdm import tqdm
# Load train_test_split from sklearn
from sklearn.model_selection import train_test_split
import random

# In[35]:


TRAIN = True


# In[36]:
# root = "/Volumes/T7/gravitational-waves/kaggle-data"
root = "/media/viktor/T7/gravitational-waves"


def load_trained_files():
    root = "/media/viktor/T7/gravitational-waves"
    TAKE_N = 50
    signal_files  =  glob.glob(f'{root}/generated-signals/signals_true_mix/*.npy')[:TAKE_N]
    noise_files = glob.glob(f'{root}/generated-signals/signals_noise/*.npy')[:TAKE_N]
    
    df = pd.DataFrame({'filename': signal_files + noise_files, 'target': [1] * TAKE_N + [0] * TAKE_N})
    return df
    
    


def load_df(train=True):
    
    if train:
        # Load /Users/viktorcikojevic/Documents/gravitational-waves-detection-kaggle/train_labels.csv as pandas dataframe
        df = pd.read_csv(f'{root}/kaggle-data/train_labels.csv')
        root_folder = f"{root}/kaggle-data/train"
        # Rename the column 'id' to 'filename'. Add "root_folder" to the beginning of the filename
        df['filename'] = root_folder + "/" + df['id'].astype(str) + ".hdf5"

    else:
        df = pd.DataFrame({'filename': glob.glob(f'{root}/kaggle-data/test/*.hdf5')})
    return df

df = load_df(train=TRAIN)


# # Read file function 👀

# In[37]:



# Each vector x has shape (4, 360, N). If N is not a multiple of 360, then we need to pad the vector with random noise with mu=np.average(x) and sigma=np.std(x)
def pad_amplitudes(x1, x2):
    # Get the shape of the vectors
    shape1 = x1.shape
    shape2 = x2.shape
    # Get the number of elements in each vector
    n1 = shape1[1]
    n2 = shape2[1]
    # Get the number of elements to pad
    n_pad = abs(n1 - n2)
    # Get the average and std of each vector
    mu1 = np.average(x1)
    mu2 = np.average(x2)
    sigma1 = np.std(x1)
    sigma2 = np.std(x2)
    # Get the padding vector
    pad1 = np.random.normal(mu1, sigma1, (360, n_pad))
    pad2 = np.random.normal(mu2, sigma2, (360, n_pad))
    # Pad the shorter vector
    if n1 > n2:
        x2 = np.concatenate((x2, pad2), axis=1)
    else:
        x1 = np.concatenate((x1, pad1), axis=1)
    return x1, x2



def combine_amplitudes(amplitude_0, amplitude_1):
    
    amplitude_0, amplitude_1 = pad_amplitudes(amplitude_0, amplitude_1)
    # Get the real part of the amplitudes
    real_0 = amplitude_0.real
    real_1 = amplitude_1.real
    # Get the imaginary part of the amplitudes
    imag_0 = amplitude_0.imag
    imag_1 = amplitude_1.imag
    # normalize the amplitudes to be between 0 and 1
    real_0 = (real_0 - real_0.min()) / (real_0.max() - real_0.min())
    real_1 = (real_1 - real_1.min()) / (real_1.max() - real_1.min())
    imag_0 = (imag_0 - imag_0.min()) / (imag_0.max() - imag_0.min())
    imag_1 = (imag_1 - imag_1.min()) / (imag_1.max() - imag_1.min())
    # Expand the dims of the real and imaginary parts
    real_0 = np.expand_dims(real_0, axis=0)
    real_1 = np.expand_dims(real_1, axis=0)
    imag_0 = np.expand_dims(imag_0, axis=0)
    imag_1 = np.expand_dims(imag_1, axis=0)
    # Make a vstack of the amplitudes
    amplitudes = np.vstack((real_0, imag_0, real_1, imag_1))
    return amplitudes



# Idea from this function takes from this notebook (😇): https://www.kaggle.com/code/ayuraj/g2net-understand-the-data
def read_data(file):
    file = Path(file)
    with h5py.File(file, "r") as f:
        filename = file.stem
        f = f[filename]
        h1 = f["H1"]
        l1 = f["L1"]
        freq_hz = list(f["frequency_Hz"])
        
        h1_stft = h1["SFTs"][()]
        h1_timestamp = h1["timestamps_GPS"][()]
        # H2 data
        l1_stft = l1["SFTs"][()]
        l1_timestamp = l1["timestamps_GPS"][()]
        
        return combine_amplitudes(h1_stft, l1_stft)


# In[38]:


x = read_data(df.iloc[0]['filename'])


# In[39]:


# Each vector x has shape (4, 360, N). If N is not a multiple of 360, then we need to pad the vector with random noise with mu=np.average(x) and sigma=np.std(x)
def pad_vector(x):
    if x.shape[2] % 360 != 0:
        mu = np.average(x)
        sigma = np.std(x)
        padding = np.random.normal(mu, sigma, (4, 360,360 - x.shape[2] % 360))
        x = np.concatenate((x, padding), axis=2)
    return x


# In[40]:


# Each vector x has shape (4, 360, N). Loop over the vector and slice it into vectors of shape (4, 360, 360). If N is not a multiple of 360, then we need to pad the vector with random noise with mu=np.average(x) and sigma=np.std(x) before slicing
# Return a list of vectors
def slice_vector(x):
    x = pad_vector(x)
    slices = []
    for i in range(x.shape[2] // 360):
        slices.append(x[:, :, i * 360 : (i + 1) * 360])
    return np.array(slices)


# In[41]:


# For a given vector (M, 4, 360, 360) and model, return the predictions of the model

def get_predictions(x, model):
    x = tf.convert_to_tensor(x)
    return model(x)
    predictions = []
    for i in range(x.shape[0]):
        predictions.append(model.predict(x[i].reshape(1, 4, 360, 360)))
    return np.array(predictions)


# In[42]:



def get_model(path):
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
    model.load_weights(path)
    return model


# In[43]:


model_paths = {
    "mix-gamma-4": f"{root}/trained-models/run-20221030_154453-3ua05fk1/files/model-best.h5",
    # "h0=0.1": f"{root}/trained-models/run-20221030_154818-4lrmhtxo/files/model-best.h5",
    # "h0=0.2": f"{root}/trained-models/run-20221030_154903-sa3eoy63/files/model-best.h5",
    # "mix-gamma-4-more-data": f"{root}/trained-models/run-20221101_220800-1sei79yc/files/model-best.h5",
    # "mix-gamma-2": f"{root}/trained-models/run-20221101_220810-3ao4yzcd/files/model-best.h5",
    # "h0=0.1-w1": f"{root}/trained-models/run-20221101_220814-37vvoies/files/model-best.h5",
    # "h0=0.2-w1": f"{root}/trained-models/run-20221101_220818-1gznbinr/files/model-best.h5",
}


# In[44]:


# For each model in model_paths, load the model and save it to the models dictionary    
models = {}
for key, path in tqdm(model_paths.items()):
    models[key] = get_model(path)


# In[45]:


models


# In[46]:


# For each model in models and file path to the file, return the predictions of the model
def get_predictions_for_file(file, models):
    x = read_data(file)
    x = slice_vector(x)
    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = get_predictions(x, model)
    return predictions


# In[47]:


# For each file in pandas dataframe, return the predictions of the model. Name the column with the model name
import gc
def get_predictions_for_df(df, models):
    predictions = {}
    for model_name, model_path in models.items():
        predictions[model_name] = []
    for file in tqdm(df['filename']):
        # print(f"Analyzing {file} ... ")
        predictions_for_file = get_predictions_for_file(file, models)
        for model_name, model_path in models.items():
            predictions[model_name].append(predictions_for_file[model_name])
        gc.collect()
    return predictions


# In[48]:


def process_predictions(model_name, predictions):
    predictions = predictions[model_name]
    # Get the average of the predictions, std, min, max and median
    
    avg = []
    std = []
    min_ = []
    max_ = []
    median = []
    n = []
    m = []
    for p in predictions:
        avg.append(np.average(p))
        std.append(np.std(p))
        min_.append(np.min(p))
        max_.append(np.max(p))
        median.append(np.median(p))
        # Get the number of p that are greater than 0.5
        n.append(np.sum(p > 0.5))
        # Get the number of p that are less than 0.5
        m.append(np.sum(p < 0.5))
        # Return a dictionary with the above values. Keys have the format "{model_name}_{statistic}"
    return {
        f"{model_name}_avg": avg,
        f"{model_name}_std": std,
        f"{model_name}_min": min_,
        f"{model_name}_max": max_,
        f"{model_name}_median": median,
        f"{model_name}_n": n,
        f"{model_name}_m": m
    }


# In[49]:


def process_predictions_for_df(df, models, indx, train=True):
    # deep copy df to df_new
    df_new = df.copy(deep=True)
    # print(f"Analyzing index {indx} ... ")
    predictions = get_predictions_for_df(df_new, models)
    # return predictions
    for model_name, model_path in models.items():
        df_new = df_new.join(pd.DataFrame(process_predictions(model_name, predictions)))
    if train:
        df_new.to_csv(f"predictions/train/predictions-{indx}.csv", index=False)
    else:
        df_new.to_csv(f"predictions/test/predictions-{indx}.csv", index=False)
    
    
process_predictions_for_df(df[:20], models, 0, train=TRAIN)
# for i in tqdm(range(0, len(df), 5)):
#     process_predictions_for_df(df[i:i+5], models, i, train=TRAIN)
#     gc.collect()


# # Paralelize the above code
# from joblib import Parallel, delayed
# import multiprocessing
# num_cores = multiprocessing.cpu_count()
# results = Parallel(n_jobs=4)(delayed(process_predictions_for_df)(df[i:i+2], models, i) for i in tqdm(range(0, len(df), 2)))



# In[ ]:





# In[ ]:






# Rewrite the code above as function, so that it can be used for both train and test data. Input is the command line argument, which is either "train" or "test". The code should be able to handle both cases.