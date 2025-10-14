"""This script demodulates the baseband IQ signal and plays the audio."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from loading import load_iq_complex

# Constants for visualization
PHASE_PLOT_SAMPLES = 2000  # Number of samples to display in phase plot


def plot_phases(wrapped: np.ndarray, unwrapped: np.ndarray, t: np.ndarray) -> None:
    """Plot the wrapped vs unwrapped phase plots."""
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(t[:PHASE_PLOT_SAMPLES], wrapped[:PHASE_PLOT_SAMPLES])
    axs[0].set_title('Wrapped Phase')
    axs[0].set_ylabel('Phase (radians)')
    axs[0].set_xlabel('Time (ms)')
    axs[0].grid(True)

    axs[1].plot(t[:PHASE_PLOT_SAMPLES], unwrapped[:PHASE_PLOT_SAMPLES])
    axs[1].set_title('Unwrapped Phase')
    axs[1].set_ylabel('Phase (radians)')
    axs[1].set_xlabel('Time (ms)')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demodulate FM IQ signals to audio')
    parser.add_argument('filepath', help='Path to the .wav file containing IQ data')
    parser.add_argument('-o', '--output', default='demod_audio.wav', help='Output audio file path (default: demod_audio.wav)')
    parser.add_argument('-k', '--freq-dev', type=float, default=75e3, help='Frequency deviation in Hz (default: 75000)')
    args = parser.parse_args()

    # load in the IQ data as complex signal
    fs, Ts, r = load_iq_complex(args.filepath)

    # get the phase of the signal - unwrap the phase!
    phi = np.angle(r)
    phi_unwrapped = np.unwrap(phi)
    t = np.arange(len(phi)) * Ts * 1e3  # Time axis for plotting (ms)
    plot_phases(phi, phi_unwrapped, t)

    # use the first order difference to approximate the derivative
    m = (phi_unwrapped[1:] - phi_unwrapped[:-1]) / (2 * np.pi * args.freq_dev * Ts)

    # scale to (-1, 1)
    m = m / np.max(np.abs(m))

    # scale up and cast to int16 for saving
    audio_int = np.int16(m * 32767)

    # save the file
    wavfile.write(args.output, fs, audio_int)
    print(f'\nDemodulated audio saved to: {args.output}')