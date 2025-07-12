"""This script loads in and inspect the .wav IQ data."""

# imports
from  scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

dir = 'IQ/2025_07_12/'
file = '15-26-26_103900000Hz.wav'

# load the .wav file
fs, data = wavfile.read(dir + file)
Ts = 1/fs

print(f'\nSampling rate:       {fs/1e6} MHz')
print(f'Sample duration:     {Ts*1e6} us')
print(f'Length of recording: {Ts * len(data):.3f} s')

# extract the IQ data
I, Q = data[:, 0], data[:, 1]

# check the number of unique values - full utilization would be 2^16 = 65536
# - increase RF Gain slider to improve utilization
# - decrase RF Gain slider to remedy clipping
print(f'\nUnique I values: {len(np.unique(I))}')
print(f'Unique Q values: {len(np.unique(Q))}')

# plot the first 1000 samples of the data
t = Ts * np.arange(500) * 1e3  # time axis
plt.figure()
plt.plot(t, I[:500], label = 'I')
plt.plot(t, Q[:500], label = 'Q')
plt.title('FM Baseband IQ Data')
plt.xlabel('Time (ms)')
plt.legend()

# plot a histogram of the I data to see if clipping is taking place
plt.figure()
plt.hist(I, bins=512, log=True)
plt.title('I Component Histogram')
plt.xlabel('Sample values')
plt.ylabel('Value counts (log scale)')

plt.show()