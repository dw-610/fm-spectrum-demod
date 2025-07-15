"""This script demodulates the baseband IQ signal and plays the audio."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from loading import load_data


def plot_phases(wrapped, unwrapped, t):
    """Plot the wrapped vs unwrapped phase plots."""
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(t[:2000], wrapped[:2000])
    axs[0].set_title('Wrapped Phase')
    axs[0].set_ylabel('Phase (radians)')
    axs[0].set_xlabel('Time (ms)')
    axs[0].grid(True)

    axs[1].plot(t[:2000], unwrapped[:2000])
    axs[1].set_title('Unwrapped Phase')
    axs[1].set_ylabel('Phase (radians)')
    axs[1].set_xlabel('Time (ms)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__=="__main__":

    dir = 'IQ/2025_07_12/'
    file = '15-26-26_103900000Hz.wav'
    fc = 103.9e6  # Center frequency of the recording
    kf = 75e3  # Typical freq dev for FM

    # load in the IQ data
    fs, Ts, data = load_data(dir + file)

    # split into IQ components and the combine into the complex signal
    I, Q = data[:, 0], data[:, 1]
    r = I + 1j * Q

    # get the phase of the signal - unwrap the phase!
    phi = np.angle(r)
    phi_unwrapped = np.unwrap(phi)
    t = np.arange(len(phi)) * Ts * 1e3  # Time axis for plotting (ms)
    plot_phases(phi, phi_unwrapped, t)
    
    # use the first order difference to approximate the derivative
    m = (phi_unwrapped[1:] - phi_unwrapped[:-1]) / (2 * np.pi * kf * Ts)

    # scale to (-1, 1)
    m = m / np.max(np.abs(m))

    # scale up and cast to int16 for saving
    audio_int = np.int16(m * 32767)

    # save the file
    wavfile.write('test_sounds/demod_audio.wav', fs, audio_int)