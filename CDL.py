import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Butterworth
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Parameter settings
fs = 50.0  # Sampling frequency
cutoff = 1.0  # Cutoff frequency

# 滤波
filtered_signal = highpass_filter(signal, cutoff, fs)

# Plot the original signal and the filtered signal
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal, label='Original Signal')
plt.title('Original Signal with Low-Frequency Noise')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal', color='orange')
plt.title('Signal after High-Pass Filtering')
plt.legend()

plt.tight_layout()
plt.show()

