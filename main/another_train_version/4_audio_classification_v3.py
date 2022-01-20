import os
import shutil
import numpy as np
from numpy.lib.financial import rate
from scipy.fft._pocketfft.basic import fft2

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio
import librosa 
import soundfile as sf
import matplotlib.pyplot as plt
import json
from scipy.io import wavfile
import scipy.signal
import json
from pathlib import Path
import argparse
from test_noisy_v3 import audio_classification_predict

# dataset directory / ckpt model / save path /SAMPLING_RATE
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='ckpt/keras_model_noisy_v3_v1.h5',
                        help='model file to make predictions')
    parser.add_argument('--dataset_dir', type=str, default='/home/kms/nlp/dataset/test',
                        help='directory containing wavfiles to predict')
    #parser.add_argument('--noise_dir', type=str, default='./data_noise/drone_siren_concat_v3.wav',
    #                    help='directory containing noise to predict')
    parser.add_argument('--noise_dir', type=str, default='./data_noise/drone_noise_only_ch1.wav',
                        help='directory containing noise to predict')
    parser.add_argument('--save_path', type=str, default='./results_json',
                        help='time in seconds to sample audio')
    parser.add_argument('--SAMPLING_RATE', type=int, default=48000, # 추후 48000으로 수정 
                        help='sample rate of audio')
    args, _ = parser.parse_known_args()

    audio_classification_predict(args)


'''
dataset path

dataset_dir 
    - set01
        - set01_drone01_ch1.wav
        ...
    - set02
        - set02_drone01_ch1.wav
        ...
    - set03
        - set02_drone01_ch1.wav
        ...
    - set04
        - set02_drone01_ch1.wav
        ...
'''