"""This script examines the spectrum of the signal."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from loading import load_data

# Constants for spectrum analysis
SEGMENT_LENGTH = 8192  # FFT length for waterfall chunks
NOISE_FLOOR_DB = -60  # Minimum dB level for visualization
EPSILON = 1e-12  # Small value to prevent log(0)


def get_mag_spectrum(r: np.ndarray) -> np.ndarray:
    """Returns the magnitude spectrum of a time-domain sampled signal."""
    return np.abs(np.fft.fftshift(np.fft.fft(r))) / len(r)


def segment_signal(r: np.ndarray, L: int) -> np.ndarray:
    """Splits the signal into chunks of length L.

    Uses numpy stride tricks to create a sliding window view without copying data.
    This is memory efficient for large signals.
    """
    starts = np.arange(0, len(r) - L + 1, L)
    return np.lib.stride_tricks.as_strided(
        r, shape=(len(starts), L),
        strides=(r.strides[0] * L, r.strides[0])
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Examine the spectrum of FM IQ signals')
    parser.add_argument('filepath', help='Path to the .wav file containing IQ data')
    parser.add_argument('center_freq', type=float, help='Center frequency of the recording in Hz (e.g., 103.9e6)')
    args = parser.parse_args()

    # load in the IQ data
    fs, Ts, data = load_data(args.filepath)

    # split into IQ components and the combine into the complex signal
    I, Q = data[:, 0], data[:, 1]
    r = I + 1j * Q

    # normalize to [-1, 1]
    r = r / np.abs(r).max()

    # use the FFT to get the magnitude spectrum
    R = get_mag_spectrum(r)

    # define the frequency axes (\omega = 2*\pi*f*Ts)
    w = np.linspace(-np.pi, np.pi, len(R))
    f = w / (2 * np.pi * Ts) + args.center_freq  # adjust for center frequency

    # plot the magnitude spectrum of the full signal
    plt.figure(figsize=(10, 5))
    plt.plot(f / 1e6, 20 * np.log10(R))
    plt.title(f'Magnitude Spectrum of the Recording ($f_c = {args.center_freq / 1e6:.1f}$ MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('$|R(f)|$')
    plt.grid()

    # split the signal into smaller chunks
    rcs = segment_signal(r, SEGMENT_LENGTH)

    # loop through and get the magnitude spectrum of each chunk
    Rcs = np.zeros(rcs.shape)
    for i, rc in enumerate(rcs):
        Rcs[i] = get_mag_spectrum(rc)

    # convert to dB for a better visualization
    Rcs_db = 20 * np.log10(Rcs + EPSILON)

    # remove low values for better visualization
    Rcs_db = np.where(Rcs_db > NOISE_FLOOR_DB, Rcs_db, NOISE_FLOOR_DB)

    # define axes
    w = np.linspace(-np.pi, np.pi, rcs.shape[1])
    f = w / (2 * np.pi * Ts) + args.center_freq  # adjust for center frequency
    t = np.arange(rcs.shape[0]) * (rcs.shape[1] * Ts)

    # plot as a heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(Rcs_db, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[f[0] / 1e6, f[-1] / 1e6, t[0], t[-1]])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time (s)')
    plt.title('Spectrogram of the Recording (Waterfall Plot)')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()