import os
import pandas as pd
import numpy as np

# List files in generated-data/signals and generated-data/noise
signals = os.listdir("generated-data/signals")
noise = os.listdir("generated-data/noise")
# Apply absolute path
signals = [os.path.abspath(f"generated-data/signals/{f}") for f in signals]
noise = [os.path.abspath(f"generated-data/noise/{f}") for f in noise]
labels = [0] * len(noise) + [1] * len(signals)
files = noise + signals
# Create a dataframe with the files and labels
df = pd.DataFrame({"file": files, "label": labels})
# File is of type: '/media/viktor/T7/gravitational-waves-kaggle-2022/datasets/dataset-v1/generated-data/noise/amplitudes_50ab23095_22.353759768757524_False.npy'
# In this case, the depth is 22.353759768757524. 
# Create a new column with the depth
df["depth"] = df["file"].apply(lambda x: float(x.split("_")[-2]))
# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)
# save the dataframe
df.to_csv("data.csv", index=False)



# test: try to load each file
from tqdm import tqdm
for f in tqdm(df["file"].tolist()):
    # load the data. If it fails, print the file name
    try:
        np.load(f)
    except:
        # delete the file from the dataframe
        df = df[df["file"] != f]

            
df.to_csv("data.csv", index=False)
