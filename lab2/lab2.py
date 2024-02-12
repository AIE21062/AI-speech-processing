#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install librosa


# In[4]:


from glob import glob
audio_files = glob('AI.wav')


# In[5]:


import IPython.display as ipd

ipd.Audio(audio_files[0])


# In[17]:


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Function to calculate finite difference for the first derivative
def calculate_first_derivative(signal):
    diff = np.diff(signal)
    derivative_signal = np.concatenate(([diff[0]], diff))
    return derivative_signal

# Load the speech signal using librosa
file_path = 'AI.wav'
original_signal, sr = librosa.load(file_path, sr=None)

# Calculate the first derivative
derivative_signal = calculate_first_derivative(original_signal)

# Plot and visualize the original and derivative signals
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
librosa.display.waveshow(original_signal, sr=sr)
plt.title('Original Speech Signal')

plt.subplot(2, 1, 2)
librosa.display.waveshow(derivative_signal, sr=sr)
plt.title('First Derivative Signal')

plt.tight_layout()
plt.show()

# Save the derivative signal to a new audio file
derivative_file_path = 'save_derivative_audio.wav'
sf.write(derivative_file_path, derivative_signal, sr)

# Listen to the original and derivative signals
import IPython.display as ipd
print("Listening to the original speech signal:")
#ipd.Audio(file_path)

print("Listening to the first derivative signal:")
ipd.Audio(derivative_file_path)


# In[9]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# Function to calculate finite difference for the first derivative
def calculate_first_derivative(signal):
    diff = np.diff(signal)
    derivative_signal = np.concatenate(([diff[0]], diff))
    return derivative_signal

# Function to detect zero crossings in a signal
def zero_crossings(signal):
    return np.where(np.diff(np.sign(signal)))[0]

# Load the speech signal using librosa
file_path = 'AI.wav'
original_signal, sr = librosa.load(file_path, sr=None)

# Calculate the first derivative
derivative_signal = calculate_first_derivative(original_signal)

# Detect zero crossings in the first derivative signal
zero_crossings_indices = zero_crossings(derivative_signal)

# Plot and visualize the original signal, first derivative, and zero crossings
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
librosa.display.waveshow(original_signal, sr=sr)
plt.title('Original Speech Signal')

plt.subplot(3, 1, 2)
librosa.display.waveshow(derivative_signal, sr=sr)
plt.title('First Derivative Signal')

plt.subplot(3, 1, 3)
plt.plot(derivative_signal, label='First Derivative Signal')
plt.plot(zero_crossings_indices, derivative_signal[zero_crossings_indices], 'ro', label='Zero Crossings')
plt.title('Zero Crossings in First Derivative Signal')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate the average length between consecutive zero crossings for speech and silence regions
speech_indices = zero_crossings_indices[::2]  # Assuming speech regions are even indices
silence_indices = zero_crossings_indices[1::2]  # Assuming silence regions are odd indices

average_length_speech = np.mean(np.diff(speech_indices))
average_length_silence = np.mean(np.diff(silence_indices))

print("Average length between consecutive zero crossings in speech regions:", average_length_speech)
print("Average length between consecutive zero crossings in silence regions:", average_length_silence)


# In[18]:


import librosa

def audio_length(file_path):
    signal, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=signal, sr=sr)
    return duration


MY_audio_path = 'OKFINE.wav'
teammate_audio_path = 'speech1.wav'

MY_audio_length = audio_length(MY_audio_path)
teammate_audio_length = audio_length(teammate_audio_path)

print("MY spoken word length:", MY_audio_length, "seconds")
print("Teammate's spoken word length:", teammate_audio_length, "seconds")


# In[19]:


import librosa
import numpy as np

def get_pitch(audio_file_path):
    # Load the audio file
    y, sr = librosa.load(audio_file_path)

    # Compute the pitch using Harmonic-Percussive Source Separation
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Average pitch across time frames
    avg_pitch = np.mean(pitches)

    return avg_pitch

# Replace with actual file paths
question_pitch = get_pitch('question.wav')
statement_pitch = get_pitch('statement.wav')

# Compare pitch characteristics
print("Pitch analysis results:")
print(f"Question average pitch: {question_pitch:.2f} Hz")
print(f"Statement average pitch: {statement_pitch:.2f} Hz")


# In[ ]:




