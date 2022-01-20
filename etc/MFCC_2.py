import scipy.signal as signal
import math
import librosa
import numpy as np

def wav_fft(file_name):
    print("fft start")
    audio_sample, sampling_rate = librosa.load(file_name, sr = None)        
    fft_result = librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann).T                    
    mag, phase = librosa.magphase(fft_result)                    
    print("fft end")
    return mag, sampling_rate

#normalize_function
min_level_db = -100
def _normalize(S):
    return np.clip((S-min_level_db)/(-min_level_db), 0, 1)


mag, sampling_rate = wav_fft("set01_drone01.wav")
mag_db = librosa.amplitude_to_db(mag)
mag_n = _normalize(mag_db)

import matplotlib.pyplot as plt
import librosa.display

librosa.display.specshow(mag_n.T, y_axis='linear', x_axis='time', sr=sampling_rate)
plt.title('FFT result')
plt.show()