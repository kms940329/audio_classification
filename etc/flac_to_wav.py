import soundfile as sf
import librosa
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

import seaborn as sns

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers, models
from IPython import display


import soundfile as sf

# Read annotation file
annotation_file = "/home/kms/nlp/dataset/speech/AUDIO_INFO"

file_exist = os.path.isfile(annotation_file)

try:
    df = pd.read_csv(annotation_file, sep='|', header=None)
except FileNotFoundError:
    print("==================================")
    print("annoatation file을 찾을 수 없습니다.")
    print("==================================")


# Load audio file
'''

# /home/kms/nlp/dataset/speech/train_data_01/003/106
train_file = Path("/home/kms/nlp/dataset/speech/train_data_01/003")
#train_file = "/home/kms/nlp/dataset/speech/train_data_01/003"
train_file_path = list(train_file.glob(r"*/*.flac"))
#File_Path = pd.Series(File_Path).astype(str)
print(train_file_path[0])

train_file_path[0].split('/')

print(train_file_path[0].split('/'))

'''
print(df)

# df[2][1:115]
#print("speakers : {}\n".format(len(df)-1))
#print(df[2][1:len(df)-1])

train_number = []
train_gender = []
test_number = []
test_gender = []

for i in range(1, len(df)):
    if df[4][i] == 'train_data_01': # train data
        train_number.append(df[0][i])
        train_gender.append(df[2][i])
    else: # test data
        test_number.append(df[0][i])
        test_gender.append(df[2][i])

'''
print(train_gender)
print(train_number)
print(len(train_gender))
print(len(train_number))
print("train speakers : {}".format(len(train_number)))

print(test_gender)
print(test_number)
print(len(test_gender))
print(len(test_number))
print("train speakers : {}".format(len(test_number)))
'''
# /home/kms/nlp/dataset/speech/train_data_01/003/106
train_base_path = "/home/kms/nlp/dataset/speech/train_data_01/003"
test_base_path = "/home/kms/nlp/dataset/speech/test_data_01/003"

#print(train_number[0])

# pandas truncation setting
pd.set_option('display.max_colwidth', -1)

# 리스트로 해보기
train_file_path = []
train_label = []
test_file_path = []
test_label = []

for train_j in range(0, len(train_number)):
    train_path = os.path.join(train_base_path, train_number[train_j])
    train_path = Path(train_path)
    train_file_path_tmp = list(train_path.glob(r"*.flac"))

    train_label_tmp = train_gender[train_j]
    for train_k in range(len(train_file_path_tmp)-1):
        train_file_path.append(train_file_path_tmp[train_k])
        train_label.append(train_label_tmp)
    
for test_j in range(0, len(test_number)):
    test_path = os.path.join(test_base_path, test_number[test_j])
    test_path = Path(test_path)
    test_file_path_tmp = list(test_path.glob(r"*.flac"))

    test_label_tmp = test_gender[test_j]
    for test_k in range(len(test_file_path_tmp)-1):
        test_file_path.append(test_file_path_tmp[test_k])
        test_label.append(test_label_tmp)


print(len(train_file_path))
print(len(train_label))
print(len(test_file_path))
print(len(test_label))

# train_data 
train_file_path = pd.Series(train_file_path).astype(str)
train_label = pd.Series(train_label)
train_df = pd.concat([train_file_path, train_label], axis=1)
train_df.columns = ['audio', 'label']

# test_data
test_file_path = pd.Series(test_file_path).astype(str)
test_label = pd.Series(test_label)
test_df = pd.concat([test_file_path, test_label], axis=1)
test_df.columns = ['audio', 'label']


print(len(train_df))
print(train_df)

print(len(test_df))
print(test_df)
# train_df.to_csv('pandas_tmp.txt')

#print(train_df.iloc[0:2]['audio'])
print(train_df.iloc[0:2]['label'])

save_dir = '/home/kms/nlp/dataset/m_f_c/' # save dir

audio_data, audio_sr = sf.read('/home/kms/nlp/dataset/m_f_c/children/0001_063.wav')
print(audio_sr)
'''
for i in range(0,len(train_file_path)):
    if train_df.iloc[i]['label'] == 'f':
        label_name = 'female/'
        save_path = os.path.join(save_dir, label_name)
        file_path = train_df.iloc[i]['audio']
        file_name = file_path.split('/')

        for tmp_name in file_name:
            if tmp_name.endswith('.flac'):
                file_name = tmp_name
            else:
                pass

        audio_data, audio_sr = sf.read(file_path)
        #sf.write(save_dir+label_name+file_name+'.wav', audio_data, audio_sr, format='WAV')
        print(audio_sr)

        #print(save_path)
    else:
        label_name = 'male/'
        save_path = os.path.join(save_dir, label_name)
        file_name = train_df.iloc[i]['audio']
        #print(save_path)

        file_path = train_df.iloc[i]['audio']
        file_name = file_path.split('/')

        for tmp_name in file_name:
            if tmp_name.endswith('.flac'):
                file_name = tmp_name
            else:
                pass

        audio_data, audio_sr = sf.read(file_path)
        #sf.write(save_dir+label_name+file_name+'.wav', audio_data, audio_sr, format='WAV')
        print(audio_sr)


'''



'''
data_dir = 
save_dir = '/home/kms/nlp/dataset/m_f_c/' # save dir

w_flac = '/mnt/junewoo/speech/TIDIGITS/tidigits_flac/data/adults/train/woman/sp/o31oa.flac' # read file

_, w_id = os.path.split(w_flac) # get dir, file name for saving

w_id = w_id[:-5] # change name o31oa.flac -> o31oa

w_data, w_sr = sf.read(w_flac) # data, sampling rate divide

sf.write(save_dir + w_id + '.wav', w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav

w, sr = librosa.load(save_dir+w_id+'.wav', sr=20000)


g_sr = 16000 # goal sampling rate

w_resample = librosa.resample(w, sr, g_sr)
m_resample = librosa.resample(m, sr, g_sr)

sf.write(save_dir + w_id + '_resample16k.wav', w_resample, g_sr, format='WAV', endian='LITTLE', subtype='PCM_16') # file write for wav
sf.write(save_dir + m_id + '_resample16k.wav', m_resample, g_sr, format='WAV', endian='LITTLE', subtype='PCM_16')
'''
