"""This script examines the spectrum of the signal."""

import numpy as np
import matplotlib.pyplot as plt
from loading import load_data


def get_mag_spectrum(r: np.ndarray):
    """Returns the magnitude spectrum of a time-domain sampled signal."""
    return np.abs(np.fft.fftshift(np.fft.fft(r))) / len(r)


def segment_signal(r, L):
    """Splits the signal into chunks of length L."""
    starts = np.arange(0, len(r) - L + 1, L)
    return np.lib.stride_tricks.as_strided(
        r, shape=(len(starts), L),
        strides=(r.strides[0] * L, r.strides[0])
    )


if __name__=="__main__":
 
    dir = 'IQ/2025_07_12/'
    file = '15-26-26_103900000Hz.wav'
    fc = 103.9e6  # Center frequency of the recording

    # load in the IQ data
    fs, Ts, data = load_data(dir + file)

    # split into IQ components and the combine into the complex signal
    I, Q = data[:, 0], data[:, 1]
    r = I + 1j * Q

    # normalize to [-1, 1]
    r = r / max(max(I), max(Q))

     # use the FFT to get the magnitude spectrum
    R = get_mag_spectrum(r)

    # define the frequency axes (\omega = 2*\pi*f*Ts)
    w = np.linspace(-np.pi, np.pi, len(R))
    f = w / (2 * np.pi * Ts) + fc  # adjust for center frequency

    # plot the magnitude spectrum of the full signal
    plt.figure(figsize=(10,5))
    plt.plot(f/1e6, 20*np.log10(R))
    plt.title('Magnitude Spectrum of the Recording ($f_c = 103.9$ MHz)')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('$|R(f)|$')
    plt.grid()
    
    # split the signal into smaller chunks
    rcs = segment_signal(r, 8192)

    # loop through and get the magnitude spectrum of each chunk
    Rcs = np.zeros(rcs.shape)
    for i, rc in enumerate(rcs):
        Rcs[i] = get_mag_spectrum(rc)

    # convert to dB for a better visualization
    Rcs_db = 20 * np.log10(Rcs + 1e-12)

    # # remove low values for better visualization
    Rcs_db = np.where(Rcs_db > -60, Rcs_db, -60)

    # define axes
    w = np.linspace(-np.pi, np.pi, rcs.shape[1])
    f = w / (2 * np.pi * Ts) + fc  # adjust for center frequency
    t = np.arange(rcs.shape[0]) * (rcs.shape[1] * Ts)

    # plot as a heatmap
    plt.figure(figsize=(12,8))
    plt.imshow(Rcs_db, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[f[0]/1e6, f[-1]/1e6, t[0], t[-1]])
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Time (s)')
    plt.title('Spectrogram of the Recording (Waterfall Plot)')
    plt.colorbar(label='Magnitude (dB)')
    plt.show()
    