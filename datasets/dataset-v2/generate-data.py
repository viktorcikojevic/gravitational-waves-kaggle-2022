
import numpy as np
import glob 
import pandas as pd
import h5py # import to read hdf5
from pathlib import Path
import pyfstat
from scipy import stats
import os
from joblib import Parallel, delayed
import shutil
from tqdm import tqdm
import sys 

# root = "/Volumes/T7/gravitational-waves/kaggle-data"
root = "/media/viktor/T7/gravitational-waves-kaggle-2022"

def load_trained_files(train=True):
   if train:
      df = pd.read_csv(f'{root}/kaggle-data/train_labels.csv')
      df['filename'] = f'{root}/kaggle-data/train/' + df['id'].astype(str) + ".hdf5"
   else:
      test_files = glob.glob(f'{root}/kaggle-data/test/*.hdf5')
      df = pd.DataFrame({'filename': test_files})
      
   return df
print("[INFO] Loading files ...")
df_train = load_trained_files(train=False)
# df_test = load_trained_files(train=True)
# concat the dataframes
# df_data = pd.concat([df_train, df_test], axis=0)
df_data = df_train
print("[INFO] Files loaded ...")

# Switch root to the current directory
root = os.getcwd()


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
        
        return (h1_timestamp, l1_timestamp), freq_hz




def generate_signals(file, h0, depth, start, h1_timestamp, l1_timestamp, frequencies, draw_signal, seed):
    # Returns the sampled signal.
    # Parameters:
    # t_start: start time of the signal
    # n_time_segments: number of time segments to sample
    # detector: detector to sample from, it can be "H1" or "L1"
    # sqrtSX: noise level
    # F0: median of frequency distribution
    # h0: GW strain. It is set to 0 if draw_signal is False
    # draw_signal: if True, the signal is drawn from a distribution. If False, the signal is set to 0.
    
    
    
    # These parameters describe background noise and data format
    writer_kwargs = {
                    # "tstart": t_start, # initial start time of the data
                    # "duration": int(1800 * n_time_segments),
                    "detectors": "H1,L1",        
                    "sqrtSX": h0 * depth, # noise level
                    "Tsft": 1800,            
                    "Band": 359/1800 , #  
                    "SFTWindowType": "tukey", 
                    "SFTWindowBeta": 0.01,
                    "timestamps": {"H1": h1_timestamp, "L1": l1_timestamp}
                    }

    # This class allows us to sample signal parameters from a specific population.
    # Implicitly, sky positions are drawn uniformly across the celestial sphere.
    # PyFstat also implements a convenient set of priors to sample a population
    # of isotropically oriented neutron stars.
    f0_avg = np.average(frequencies) + 0.00027777777779647295
    
    
    signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
        priors={
            # "tref": writer_kwargs["tstart"],
            # "timestamps": writer_kwargs["timestamps"],
            # "F0": {"uniform": {"low": 50, "high": 500}},
            # "F0": np.random.randint(50,500),
            "F0": np.average(frequencies) + 0.00027777777779647295, #0.00027777777779647295, # central GW frequency            
            "F1": lambda: min(10**np.random.default_rng(seed).uniform(-12, -8),0.04/(h1_timestamp[-1]-h1_timestamp[0])), # GW frequency derivative
            "F2": 0,
            "h0": h0 if draw_signal else 0, # GW strain
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
        },
        seed=seed
    )
    
    

    params = signal_parameters_generator.draw()

    DIRECTORY_TMP = f"{root}/generated-data/TMP/signals_{file}_{h0}_{depth}_{draw_signal}"
    # if os.path.exists(DIRECTORY_TMP):
    #     shutil.rmtree(DIRECTORY_TMP)

    
    
    writer_kwargs["outdir"] = DIRECTORY_TMP

    writer = pyfstat.Writer(**writer_kwargs, **params)
    
    # delete file /media/viktor/T7/gravitational-waves/generated-data/signals_ec138b0f2_7_17.50190933209334_L1_True/PyFstat.cff
    # if os.path.exists(f"{DIRECTORY_TMP}/PyFstat.cff"):
    #     os.remove(f"{DIRECTORY_TMP}/PyFstat.cff")
    
    writer.make_data(verbose=False)
    
    # return

    
    # return writer.sftfilepath
    
    # Data can be read as a numpy array using PyFstat
    frequency, timestamps, amplitudes = pyfstat.utils.get_sft_as_arrays(writer.sftfilepath)
    
    
    
    print("SHAPE \n\n", amplitudes["H1"].shape, "\n\n")
    npy_directory = f"{root}/generated-data/signals" if draw_signal else f"{root}/generated-data/noise"
    
    def preprocess_amplitude(x):        
        
        x = x[:, start:start+4096] 
        
        # Make x.real go from -1 to 1
        x.real = x.real / 1.E-25
        x.imag = x.imag / 1.E-25
        
        
        # # if the signal is too short, we pad it with zeros
        # if x.shape[1] < 4096:
        #     x = np.pad(x, ((0,0),(0,4096-x.shape[1])), 'constant')
        
        # x is currently of shape (360, 4096)
        x = x.reshape((360, 256,16))
        
        x = np.abs(x)
        
        # max pooling along the axis = 2
        x = np.max(x, axis=2)
        # x is currently of shape (360, 256)
        
        
        
        # print("max of x", np.max(x))
        
        return np.squeeze(255 * x / np.max(x))
    
    amplitudes["H1"] = preprocess_amplitude(amplitudes["H1"])
    amplitudes["L1"] = preprocess_amplitude(amplitudes["L1"])
    
    # make amplitudes between 0 and 1
    amplitudes_h1 = amplitudes["H1"]
    amplitudes_l1 = amplitudes["L1"]
    
    # stack amplitudes from both detectors
    amplitudes = np.stack((amplitudes_h1, amplitudes_l1), axis=2)
    amplitudes = np.squeeze(amplitudes)
    
    # random roll along the axis = 1
    # amplitudes = np.roll(amplitudes, int(np.random.default_rng(seed).uniform(0, 60)), axis=1)
    
    # check if there's any nan in amplitudes. If there is print it
    if np.isnan(amplitudes).any():
        raise Exception("There is a nan in amplitudes")
    
    print("[INFO] Saving data to npy file: ", f"{npy_directory}/amplitudes_{file}_{depth}_{draw_signal}.npy")
    np.save(f"{npy_directory}/amplitudes_{file}_{depth}_{draw_signal}.npy", amplitudes)
    
    
    # shutil.rmtree(DIRECTORY_TMP)
    os.system(f"rm {DIRECTORY_TMP}/*sft*")
    # os.system(f"rm {DIRECTORY_TMP}/*csv*")




def generate_signals_for_file(file, seed=42):
    x, freq_hz = read_data(file)
    h1_timestamp, l1_timestamp = x
    # 2.013696785246477e-24 54.78307 235.3751 215.0395
    # 4.232138524931458e-25 4.251791 26.257027 26.709515

    # (19 - 26) / (1.5 - 0.8) * (0.1 - 0.8) + 26 = 33
    # (19 - 26) / (1.5 - 0.8) * (1. - 0.8) + 26 = 24
    # (19 - 26) / (1.5 - 0.8) * (1.2 - 0.8) + 26 = 22
    
    h0 = 1. * np.random.uniform(0.1, 1.2) # 10**np.random.uniform(-25, -22)
    depth = (19 - 26) / (1.5 - 0.8) * (h0 - 0.8) + 26
    h0 = h0 * 1.E-26
    start_h1 = 0 # int(np.random.uniform(0, len(h1_timestamp) - 4096))
    start_l1 = 0 # min(start_h1, len(l1_timestamp) - 4096)
    # start_l1 = int(np.random.uniform(0, len(l1_timestamp) - 4096))
    # '/media/viktor/T7/gravitational-waves/kaggle-data/test/00054c878.hdf5' -> '00054c878'
    file = file.split("/")[-1].split(".")[0]
    print(f"\n\nGenerating signal for {file} with seed {seed} and detector H1. \n\n")
    
    
    # Generate signals
    generate_signals(file,h0, depth, start_h1, h1_timestamp,l1_timestamp, freq_hz, True, seed)
    
    # Generate noise
    generate_signals(file, h0, depth, start_h1, h1_timestamp,l1_timestamp, freq_hz, False, seed)
# Loop over all df_data file
files = df_data["filename"].tolist() * 2000 # 
# shuffle files 
# seed is 1st argument to the script
seed = int(sys.argv[1])
np.random.default_rng(seed).shuffle(files)
# 1.5 minutes for 128 files. For 128000 files it will take: 128000/128 * 1.5 = 1500 minutes =  25 hours
files = files[:40_000] # 1000 generates 4.7 GB of data. 700/4.7 = 148. 

npy_directory = f"{root}/generated-data/signals" 
# if the directory does not exist, create it
if not os.path.exists(npy_directory):
    os.makedirs(npy_directory)

npy_directory = f"{root}/generated-data/noise" 
# if the directory does not exist, create it
if not os.path.exists(npy_directory):
    os.makedirs(npy_directory)


# using joblib, loop over all files and generate signals
# import necessary libraries


# for seed, file in tqdm(enumerate(files)):
#     generate_signals_for_file(file, seed)

Parallel(n_jobs=14)(delayed(generate_signals_for_file)(file, int(190E+05 + seed)) for seed, file in tqdm(enumerate(files)))


# /home/viktor/Documents/generated-data/signals/amplitudes_0b50df53b_19.39823772253202_True.npyGenerating 