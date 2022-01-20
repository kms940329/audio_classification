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

###### gpu settting #######
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)



def audio_classification_predict(args):
    BATCH_SIZE = 1
    #SAMPLING_RATE = 16000
    SAMPLING_RATE = args.SAMPLING_RATE
    # SAMPLING_RATE = 48000
    # 다시 학습한 모델은 48000로 학습 및 테스트 

    # json save path 
    # results_save_path = './results_json'
    results_save_path = args.save_path

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)
        print("create results folder")


    """
    ## Model Definition
    """

    def residual_block(x, filters, conv_num=3, activation="relu"):
        # Shortcut
        s = keras.layers.Conv1D(filters, 1, padding="same")(x)
        for i in range(conv_num - 1):
            x = keras.layers.Conv1D(filters, 3, padding="same")(x)
            x = keras.layers.Activation(activation)(x)
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Add()([x, s])
        x = keras.layers.Activation(activation)(x)
        return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


    def build_model(input_shape, num_classes):
        
        inputs = keras.layers.Input(shape=input_shape, name="input") # (8000, 1)
        
        x = residual_block(inputs, 16, 2)
        x = residual_block(x, 32, 2)
        x = residual_block(x, 64, 3)
        x = residual_block(x, 128, 3)
        x = residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        return keras.models.Model(inputs=inputs, outputs=outputs)


    # model = build_model((SAMPLING_RATE, 1), 3)
    model = build_model((SAMPLING_RATE // 2, 1), 4)

    # model.summary()

    # Compile the model using Adam's default learning rate
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model_path = args.model_fn
    model.load_weights(model_path)
    #model.load_weights('./ckpt/keras_model_denoising_300.h5')


    """
    ## preprocessing function
    """


    # (1)
    def paths_and_labels_to_dataset(audio_paths):
        """Constructs a dataset of audios and labels."""
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x: path_to_audio(x))
        #label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds))

    def path_to_audio(path):
        """Reads and decodes an audio file."""
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
        return audio

    # (2)
    def audio_to_fft(audio):
        # Since tf.signal.fft applies FFT on the innermost dimension,
        # we need to squeeze the dimensions and then expand them again
        # after FFT
        audio = tf.squeeze(audio, axis=-1)
        #print("audio shape in audio_to_fft : {}".format(audio.shape))
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        fft = tf.expand_dims(fft, axis=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])



    """
    ## load test dataset
    """
    # folder_path = '/home/kms/nlp/dataset/test_tmp_children' # label : 3
    # folder_path = '/home/kms/nlp/dataset/test_tmp_female' # label : 0
    # folder_path = '/home/kms/nlp/dataset/test_tmp_male' # label : 1
    # folder_path = '/home/kms/nlp/dataset/test_tmp_noise'  # label : 2

    # folder_path = '/home/kms/nlp/dataset/test'
    folder_path = args.dataset_dir

    set_name = os.listdir(folder_path)

    test_files_name = []
    test_files_path = []

    for c, test_n in enumerate(set_name):
        tmp_path = os.path.join(folder_path, test_n)
        #test_files_path.append(tmp_path)
        
        tmp_file = os.listdir(tmp_path)
        #test_files_name.append(tmp_file)
        for d, e in enumerate(tmp_file):
            test_files_path_tmp = os.path.join(tmp_path, tmp_file[d])
            test_files_name.append(test_files_path_tmp)


    #test_files_name = os.listdir(folder_path) # file name.wav
    #test_files_path = os.listdir(folder_path) # full path about files

    # ch1 file 만 사용하기 
    ch1_file = []

    for file in test_files_name:
        if file.endswith('ch1.wav'):
            ch1_file.append(file)
        else:
            pass

    test_files_name = ch1_file

    for i in range(len(test_files_name)):
        test_files_path_tmp = os.path.join(folder_path, test_files_name[i])
        test_files_path.append(test_files_path_tmp)

    test_files_path.sort()

    #for i in range(len(test_files_path)):
    #    test_files_path[i] = os.path.join(folder_path, test_files_path[i])

    # using only drone noise without siren
    # noise_path = './data_v2/sample_noise_concat.wav'
    # noise_path = '/home/kms/nlp/NLP_project_v3/data_noise/sample_concat_ch1.wav'
    noise_path = args.noise_dir

    # data_noise, rate_noise = librosa.load(noise_path, mono=True, sr=48000, offset=0)
    data_noise, rate_noise = librosa.load(noise_path, mono=True, sr=SAMPLING_RATE, offset=0)

    #print("number of test video set : {}".format(len(test_files_path)))

    #for k, test_file_name in enumerate(test_files_path):
    #    print("{}.test file list : {}".format(k, test_file_name))
    #print("\n")
    # audio = path_to_audio(test_files_path[i])


    # folder_path = '/home/kms/nlp/dataset/test_tmp_children' # label : 3
    # folder_path = '/home/kms/nlp/dataset/test_tmp_female' # label : 0
    # folder_path = '/home/kms/nlp/dataset/test_tmp_male' # label : 1
    # folder_path = '/home/kms/nlp/dataset/test_tmp_noise'  # label : 2

    #probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])



    for count, file in enumerate(test_files_path):
        #print(file)
        #print(file.split('/')[-1])
        file_name = file.split('/')[-1]
        #print(file_name.split('_'))
        current_set_name = file_name.split('_')[0]
        current_drone_name = file_name.split('_')[1]
        #print(current_set_name, current_drone_name)
        data, rate = librosa.load(test_files_path[count], sr=SAMPLING_RATE, offset=0, mono=True) # load audio data, rate   

        M_list = []
        W_list = []
        C_list = []        

        # denoising
        #data = removeNoise(audio_clip=data, noise_clip=data_noise,verbose=False,visual=False)
        #sf.write("concat_denoising.wav" ,data, 48000, format='WAV', endian='LITTLE', subtype='PCM_16')
        for count_many in range(0, (len(data)//SAMPLING_RATE)):
            #print(count_many)
            start = SAMPLING_RATE*count_many
            end = (SAMPLING_RATE*count_many)+SAMPLING_RATE
            print("start : {}, end : {}".format(start/SAMPLING_RATE, end/SAMPLING_RATE))
            data_tmp = data[start:end].reshape(1,SAMPLING_RATE,1)
            fft = audio_to_fft(data_tmp)
            y_pred_pro = model.predict(fft)
            # Take random samples
            y_pred = np.argmax(y_pred_pro, axis=-1) # 0 :female / 1 : male / 2 : noise / 3 : children
            #print(round(np.max(y_pred_pro), 3))
            print(np.max(y_pred_pro))
            print(y_pred)