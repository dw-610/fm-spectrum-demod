"""This script filters and downsamples the raw demodulated .wav file."""

import argparse
from fractions import Fraction
from scipy.signal import butter, sosfilt, freqz_sos
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

from loading import load_data
from spectrum import get_mag_spectrum


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Filter and downsample demodulated audio')
    parser.add_argument('file', type=str, help='Path to the .wav file')
    parser.add_argument('--target-fs', type=int, default=44100,
                        help='Target sampling rate in Hz (default: 44100)')
    parser.add_argument('--filter-cutoff', type=float, default=20e3,
                        help='Lowpass filter cutoff frequency in Hz (default: 20000)')
    parser.add_argument('--filter-order', type=int, default=8,
                        help='Butterworth filter order (default: 8)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path for the downsampled audio (optional)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize audio to use full int16 range before saving')
    args = parser.parse_args()

    # load in the raw demodulated .wav audio
    fs, Ts, x = load_data(args.file)
    X = get_mag_spectrum(x)

    fs_d = args.target_fs
    Ts_d = 1 / fs_d

    # Calculate upsampling/downsampling factors
    # Find rational approximation: fs_d/fs = up/down
    frac = Fraction(fs_d, int(fs)).limit_denominator(100000)
    upsample_factor = frac.numerator
    downsample_factor = frac.denominator

    # --------------------------------------------------------------------------

    # downsample to target fs and look at the spectrum
    # upsample by calculated factors and then downsample
    x_d = np.repeat(x, upsample_factor)[::downsample_factor]
    X_d = get_mag_spectrum(x_d)

    # --------------------------------------------------------------------------

    # create a low pass filter
    sos = butter(args.filter_order, args.filter_cutoff, btype='Low', output='sos', fs=fs)
    w, h = freqz_sos(sos, fs=fs)

    # plot the frequency response
    magnitude_db = 20 * np.log10(np.abs(h))
    phase_deg = np.angle(h, deg=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,8))

    ax1.semilogx(w, magnitude_db)
    ax1.set_title('Butterworth Lowpass Filter Frequency Response')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True)
    ax1.set_xlim([10, fs/2])  # Limit to Nyquist frequency
    ax1.set_ylim([-100, 10])  # Adjust y-axis for better visibility

    ax2.semilogx(w, phase_deg)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True)
    ax2.set_xlim([10, fs/2])  # Limit to Nyquist frequency

    # filter the original signal
    x_f = sosfilt(sos, x).astype('int16')
    X_f = get_mag_spectrum(x_f)

    # --------------------------------------------------------------------------

    # downsample the filtered signal
    # upsample by calculated factors and then downsample
    x_df = np.repeat(x_f, upsample_factor)[::downsample_factor]
    X_df = get_mag_spectrum(x_df)

    # --------------------------------------------------------------------------

    # save the downsampled audio if output path is specified
    if args.output:
        # Convert to int16 for saving
        if args.normalize:
            # Normalize to use full int16 range
            x_save = x_df.astype(float)
            x_save = x_save / np.max(np.abs(x_save))  # Normalize to [-1, 1]
            x_save = (x_save * 32767).astype('int16')
        else:
            # Clip to int16 range without normalization
            x_save = np.clip(x_df, -32768, 32767).astype('int16')

        wavfile.write(args.output, fs_d, x_save)
        print(f"Saved downsampled audio to {args.output}")

    # --------------------------------------------------------------------------

    # plot all the spectrums

    w = np.linspace(-np.pi, np.pi, len(X))
    f = w / (2 * np.pi * Ts)

    w = np.linspace(-np.pi, np.pi, len(X_d))
    f_d = w / (2 * np.pi * Ts_d)

    # Combined 2x2 spectrum plots
    fig, axs = plt.subplots(2, 2, figsize=(20, 10), sharey=True)

    # Original Signal
    axs[0, 0].plot(f / 1e3, 20 * np.log10(X))
    axs[0, 0].set_xlabel('Frequency (kHz)')
    axs[0, 0].set_ylabel('$|X(f)| (dB)$')
    axs[0, 0].set_title('Original Signal')
    axs[0, 0].grid(True)
    axs[0, 0].set_ylim([-40, 50])

    # Directly Downsampled
    axs[0, 1].plot(f_d / 1e3, 20 * np.log10(X_d))
    axs[0, 1].set_xlabel('Frequency (kHz)')
    axs[0, 1].set_ylabel('$|X(f)| (dB)$')
    axs[0, 1].set_title('Directly Downsampled')
    axs[0, 1].grid(True)
    axs[0, 1].set_ylim([-40, 50])

    # Filtered Signal
    axs[1, 0].plot(f / 1e3, 20 * np.log10(X_f))
    axs[1, 0].set_xlabel('Frequency (kHz)')
    axs[1, 0].set_ylabel('$|X(f)| (dB)$')
    axs[1, 0].set_title('Filtered Signal')
    axs[1, 0].grid(True)
    axs[1, 0].set_ylim([-40, 50])

    # Downsampled after Filtering
    axs[1, 1].plot(f_d / 1e3, 20 * np.log10(X_df))
    axs[1, 1].set_xlabel('Frequency (kHz)')
    axs[1, 1].set_ylabel('$|X(f)| (dB)$')
    axs[1, 1].set_title('Downsampled after Filtering')
    axs[1, 1].grid(True)
    axs[1, 1].set_ylim([-40, 50])

    plt.tight_layout()

    plt.show()
