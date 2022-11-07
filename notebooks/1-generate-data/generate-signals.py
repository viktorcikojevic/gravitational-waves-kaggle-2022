#!/usr/bin/env python
# coding: utf-8

# 
# - About the parameters
# Two parameters for the location on the sky: Alpha, Delta
# Three parameters for the orientation of the source: cosi, psi, phi
# One parameter that describes the "strength" of the signal: h0
# Two intrinsic parameters: frequency and spin-down: F0, F1
# 
# -  FFTs
# Hi.Is there a fixed window length for fourier transform？
# 
# For this kind of signal the recommended lenght is 1800seconds (30 mins). There might be gaps in the timestamp but the considered length is 1800 seconds. Rodrigo can confirm better. :)
# 
# Yes, each timestamp labels a period of time of 1800s (you can check that by noting the frequency resolution of 1/1800 Hz), but timestamps need not to be consecutive (i.e. there may be times at which no data is collected).
# 
# - About targets 
# 
# Hey Rodrigo Tenorio, thank you for providing resources for generating more data. I am unclear about the "target" part while generating the data. The kenrel provided to generate the data does not seem to the information about the label (target 1, 0, or -1). Or am I missing something? Thanks in advance.
# 
# 0/1 labels refer to whether we included a simulated signal to that sample (1) or if it consists only on noise.
# In terms of the quantities of the kernel your refer to, a label of 1 would correspond to a signal with an amplitude h0 greater than 0 (i. e. there's a signal), while a label of 0 would correspond to not having a signal (i.e. amplitude h0 = 0).
# The same definition can be made in terms of SNR: Label of 1 corresponds to SNR > 0, label 0 corresponds to SNR = 0 (no signal added at all).

# In[22]:


import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import pyfstat

from scipy import stats
from tqdm import tqdm
import numpy as np


# In[23]:


def combine_amplitudes(amplitude_0, amplitude_1):
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
    


# In[24]:



ROOT = os.path.abspath(os.path.join(os.getcwd(), "../.."))
def _generate_signals(num_signals, tfrac, prior_index, duration, freq_min, h0, signal_rate, sqrtSX, draw_signal):
    tmin = 1238166018
    dt = 10558244
    
    # These parameters describe background noise and data format
    writer_kwargs = {
                    "tstart": int(tmin + dt*tfrac), # 1238177971, # 1238166018,
                    "duration": 1800 * min(duration, 360),
                    "detectors": "H1,L1",        
                    "sqrtSX": sqrtSX, # lambda: 10**stats.uniform(-26, -23).rvs(),          
                    "Tsft": 1800,             
                    "SFTWindowType": "tukey", 
                    "SFTWindowBeta": 0.01,
                    }

    # This class allows us to sample signal parameters from a specific population.
    # Implicitly, sky positions are drawn uniformly across the celestial sphere.
    # PyFstat also implements a convenient set of priors to sample a population
    # of isotropically oriented neutron stars.
    signal_parameters_generator = pyfstat.AllSkyInjectionParametersGenerator(
        priors={
            "tref": writer_kwargs["tstart"],
            "F0": {"uniform": {"low": freq_min, "high": freq_min+1}},
            "Band": 359/1800 ,
            "F1": lambda: np.logspace(-10, -7, 100)[np.random.randint(100)],
            "F2": 0,
            "h0": h0 if draw_signal else 0,
            **pyfstat.injection_parameters.isotropic_amplitude_priors,
        },
    )

    snrs = np.zeros(num_signals)

    for ind in range(num_signals):

        # Draw signal parameters.
        # Noise can be drawn by setting `params["h0"] = 0
        params = signal_parameters_generator.draw()
        
        DIRECTORY_TMP = f"{ROOT}/generated-signals/TMP/signals_true_{ind}_{prior_index}"  if draw_signal else f"{ROOT}/generated-signals/TMP/signals_noise_{ind}_{prior_index}"
        
        writer_kwargs["outdir"] = DIRECTORY_TMP
        
        writer = pyfstat.Writer(**writer_kwargs, **params)
        writer.make_data()
        
        paths = writer.sftfilepath.split(";")
        
        # Data can be read as a numpy array using PyFstat
        frequency, timestamps, amplitudes_0 = pyfstat.utils.get_sft_as_arrays(paths[0])
        frequency, timestamps, amplitudes_1 = pyfstat.utils.get_sft_as_arrays(paths[1])
        x = combine_amplitudes(amplitudes_0["H1"], amplitudes_1["L1"])
        
        SAVE_DIRECTORY = f"{ROOT}/generated-signals/signals_mix" if draw_signal else f"{ROOT}/generated-signals/signals_noise"
        # Create SAVE_DIRECTORY if it does not exist
        if not os.path.exists(SAVE_DIRECTORY):
            os.makedirs(SAVE_DIRECTORY)
        
        np.save(f"{SAVE_DIRECTORY}/round_{np.random.randint(100,10000000)}_{ind}_{prior_index}.npy", x)
        
        def delete_path(path):
            try:
                os.remove(path)
            except OSError:
                pass
        for path in paths:
            delete_path(path)
        
            

def generate_signals(num_signals, tfrac, prior_index, duration, freq_min, h0, signal_rate, sqrtSX, draw_signal):

                     # Generate signals with parameters drawn from a specific population

    try: # 1238166018, 1248724262)
        _generate_signals(num_signals, tfrac, prior_index, duration, freq_min, h0, signal_rate, sqrtSX, draw_signal)
    except:
        print("LOL it failed")


# In[25]:




# Load digits-from-figure.csv to get the log10freq, log10h0 columns
import pandas as pd
df = pd.read_csv(f"./digits-from-figure.csv")
df = df.sample(frac=1)
freq_min = 10**np.array(df["log10freq"])
h0 = 10**np.array(df["log10h0"])


num_priors = len(h0) # about 224
num_signals = 70 # For each prior, generate 400 signals

tfrac = np.linspace(0, 1, num=num_priors)
signal_rate =          np.logspace(np.log10(1./50), np.log10(1/10), num=num_priors) # 9.95262315, 18.47849797, 17.11328304, 15.84893192, 14.67799268, 13.59356391, 12.58925412, 11.65914401, 10.79775162, 10. 
# signal_rate = 1.E-06 * np.logspace(np.log10(1./50), np.log10(1/10), num=num_priors) # 9.95262315, 18.47849797, 17.11328304, 15.84893192, 14.67799268, 13.59356391, 12.58925412, 11.65914401, 10.79775162, 10. 

sqrtX = 1./signal_rate * h0 if np.min(signal_rate) > 1./30 else 1./np.logspace(np.log10(1./50), np.log10(1/10), num=num_priors) * h0



# In[26]:


from joblib import Parallel, delayed

# generate_dataset(train_files, indx, train_test_label="train")
Parallel(n_jobs=15)(delayed(_generate_signals)(num_signals=num_signals,
                                              tfrac=tf, 
                                              prior_index=prior_index, 
                                              duration=360, 
                                              freq_min=freq_min[prior_index],
                                              h0=h0[prior_index], 
                                              signal_rate=signal_rate[prior_index], 
                                              draw_signal=np.min(signal_rate) > 1./50,
                                              sqrtSX=sqrtX[prior_index])  for prior_index, tf in zip(range(num_priors), tfrac))


# In[ ]:




