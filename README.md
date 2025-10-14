# FM Spectrum Demodulation

Python scripts for visualizing and demodulating baseband FM radio signals from IQ data stored in .wav files.

## Overview

This project provides a complete pipeline for processing FM radio signals captured with software-defined radio (SDR) hardware. The IQ data used in this project was captured using an **RTL-SDR** receiver.

## Scripts

- **loading.py** - Load and inspect IQ data from .wav files, check ADC utilization, and visualize the time-domain signal
- **spectrum.py** - Generate frequency spectrum plots and waterfall spectrograms to see what's happening in the frequency domain
- **demod.py** - Demodulate FM signals to recover the original audio using phase differentiation
- **downsample.py** - Filter and downsample demodulated audio to standard sampling rates (e.g., 44.1 kHz) with visualization of filtering effects

## Usage

### Inspect IQ data
```bash
python loading.py <filepath>
```

### View spectrum
```bash
python spectrum.py <filepath> <center_frequency>
```
Example:
```bash
python spectrum.py recording.wav 103.9e6
```

### Demodulate to audio
```bash
python demod.py <filepath> [-o OUTPUT] [-k FREQ_DEV]
```
Example:
```bash
python demod.py recording.wav -o output.wav -k 75000
```

### Filter and downsample audio
```bash
python downsample.py <filepath> [-o OUTPUT] [--target-fs RATE] [--filter-cutoff FREQ] [--normalize]
```
Examples:
```bash
# Visualize filtering and downsampling effects
python downsample.py demod_audio.wav

# Save downsampled audio to 44.1 kHz (default)
python downsample.py demod_audio.wav -o clean_audio.wav

# Downsample to 48 kHz with normalization
python downsample.py demod_audio.wav -o clean_audio.wav --target-fs 48000 --normalize

# Custom filter cutoff frequency
python downsample.py demod_audio.wav --filter-cutoff 15000 -o output.wav
```

Run any script with `-h` or `--help` for more information.

## Typical Workflow

The scripts are designed to work together in a pipeline:

1. **Verify IQ data quality** - Use `loading.py` to check ADC utilization and ensure no clipping
   ```bash
   python loading.py recording.wav
   ```

2. **Identify FM stations** - Use `spectrum.py` to visualize the spectrum and locate stations
   ```bash
   python spectrum.py recording.wav 103.9e6
   ```

3. **Extract audio** - Use `demod.py` to demodulate the FM signal
   ```bash
   python demod.py recording.wav -o demod_audio.wav
   ```

4. **Clean and resample** - Use `downsample.py` to filter and convert to standard sample rates
   ```bash
   python downsample.py demod_audio.wav -o clean_audio.wav --normalize
   ```

## Requirements

- NumPy
- SciPy
- Matplotlib

## Installation

Install dependencies using conda:
```bash
conda install --file requirements.txt
```

Or using pip:
```bash
pip install -r requirements.txt
```