"""This script loads in and inspects the .wav IQ data."""

import argparse
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

# Constants for plotting
PLOT_SAMPLES = 500
HIST_BINS = 512


def load_data(filepath: str) -> tuple[int, float, np.ndarray]:
    """Wrapper for loading in data."""
    fs, data = wavfile.read(filepath)
    Ts = 1 / fs
    return fs, Ts, data


def load_iq_complex(filepath: str) -> tuple[int, float, np.ndarray]:
    """Load IQ data and return as complex signal.

    Returns:
        fs: Sampling frequency in Hz
        Ts: Sampling period in seconds
        r: Complex signal (I + jQ)
    """
    fs, Ts, data = load_data(filepath)
    I, Q = data[:, 0], data[:, 1]
    return fs, Ts, I + 1j * Q


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Load and inspect IQ data from .wav files')
    parser.add_argument('filepath', help='Path to the .wav file containing IQ data')
    args = parser.parse_args()

    # load the .wav file
    fs, Ts, data = load_data(args.filepath)

    print(f'\nSampling rate:       {fs / 1e6} MHz')
    print(f'Sample duration:     {Ts * 1e6} us')
    print(f'Length of recording: {Ts * len(data):.3f} s')

    # extract the IQ data
    I, Q = data[:, 0], data[:, 1]

    # check the number of unique values - full utilization would be 2^16 = 65536
    # - increase RF Gain slider to improve utilization
    # - decrease RF Gain slider to remedy clipping
    print(f'\nUnique I values: {len(np.unique(I))}')
    print(f'Unique Q values: {len(np.unique(Q))}')

    # plot the first samples of the data
    t = Ts * np.arange(PLOT_SAMPLES) * 1e3  # time axis
    plt.figure()
    plt.plot(t, I[:PLOT_SAMPLES], label='I')
    plt.plot(t, Q[:PLOT_SAMPLES], label='Q')
    plt.title('FM Baseband IQ Data')
    plt.xlabel('Time (ms)')
    plt.legend()

    # plot a histogram of the I data to see if clipping is taking place
    plt.figure()
    plt.hist(I, bins=HIST_BINS, log=True)
    plt.title('I Component Histogram')
    plt.xlabel('Sample values')
    plt.ylabel('Value counts (log scale)')

    plt.show()