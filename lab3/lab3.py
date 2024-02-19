#!/usr/bin/env python
# coding: utf-8

# In[2]:


from glob import glob
audio_files = glob('question.wav')


# In[3]:


import IPython.display as ipd
 
ipd.Audio(audio_files[0])


# In[4]:


#A1
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
file_path = 'question.wav'
original_signal, sr = librosa.load(file_path, sr=None)
plt.figure(figsize=(12, 8))
 
plt.subplot(2, 1, 1)
librosa.display.waveshow(original_signal, sr=sr)
plt.title('Original Speech Signal')
duration=librosa.get_duration(y=original_signal)
print("duration of the original audio signal",duration)


# In[5]:


import librosa
import librosa.display
import soundfile as sf
import IPython.display as ipd
 
def remove_silence(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    trimmed_signal, index = librosa.effects.trim(signal)
 
    return trimmed_signal, sr
 
 
original_audio_path = 'question.wav'
trimmed_signal, sr = remove_silence(original_audio_path)
trimmed_audio_path = 'trimmed_audio.wav'
sf.write(trimmed_audio_path, trimmed_signal, sr)
 
#print("Listening to the original audio:")
#ipd.Audio(original_audio_path)
 
print("Listening to the trimmed audio:")
ipd.Audio(trimmed_audio_path)


# In[9]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
audio = 'trimmed_audio.wav'
trim_signal, sr = librosa.load(audio, sr=None)
plt.figure(figsize=(12, 8))
 
plt.subplot(2, 1, 1)
librosa.display.waveshow(trim_signal, sr=sr)
plt.title(' trimmed Speech Signal')
duration=librosa.get_duration(y=trim_signal)
print("duration of the trimmed audio signal",duration)
print("sampling rate of trimmed audio:",sr)


# In[8]:


#A2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
 
# Load the audio file
file_path = "question.wav"
signal, sr = librosa.load(file_path, sr=None)
 
 
top_db_values = [20,25,30,35,40] 
split_signals = []
 
for top_db in top_db_values:
    split_signal = librosa.effects.split(signal, top_db=top_db)
    split_signals.append(split_signal)
 
plt.figure(figsize=(12, 12))
plt.subplot(len(top_db_values) + 1, 1, 1)
librosa.display.waveshow(signal, sr=sr)
plt.title('Original Signal')
 
for i, split_signal in enumerate(split_signals):
    plt.subplot(len(top_db_values) + 1, 1, i + 2)
    split_signal_plot = np.zeros_like(signal)
    for interval in split_signal:
        split_signal_plot[interval[0]:interval[1]] = signal[interval[0]:interval[1]]
    librosa.display.waveshow(split_signal_plot, sr=sr)
    plt.title(f'Split Signal (top_db={top_db_values[i]})')
 
    split_audio = np.concatenate([signal[interval[0]:interval[1]] for interval in split_signal])
    display(Audio(data=split_audio, rate=sr))
 
plt.tight_layout()
plt.show()


# In[ ]:


'''top_db method relies on amplitude thresholding to detect segments where the amplitude falls below a certain level.
A lower top_db value establishes a less strict threshold for amplitude, causing smaller amplitude values to be classified as
part of silent sections. As a result, more segments of the audio signal, including quieter sections, may be categorized as 
silent.
On the contrary, a higher top_db value sets a more stringent threshold for amplitude. In this case, only sections of the 
audio signal with amplitudes significantly higher than the threshold will be recognized as non-silent. 
Quieter sections or those near silence may not be identified as silent with a higher top_db.


the paper introduces a method based on the continuous average energy of the speech signal. This approach involves 
analyzing the energy levels in the signal, using a continuous average, and applying a threshold to detect and remove silence
segments.The proposed method achieved a 100% detection rate for unvoiced segments in speech signals, even in very noisy
environments with a signal-to-noise ratio (SNR) of -10 dB.
'''

