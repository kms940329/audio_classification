import librosa
from python_speech_features import mfcc

def compute_mfcc(audio_data, sample_rate):
    #audio_data = audio_data - np.mean(audio_data)
    #audio_data = audio_data / np.max(audio_data)
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.010, winstep=0.01,
                     numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat 

audio_sample, sampling_rate = librosa.load("set01_drone01.wav", sr = None)
mfcc = compute_mfcc(audio_sample, sampling_rate)

import matplotlib.pyplot as plt
import librosa.display

librosa.display.specshow(mfcc.T, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
