# FM Spectrum Demodulation

Python scripts for visualizing and demodulating baseband FM radio signals from IQ data stored in .wav files.

## Scripts

- **loading.py** - Load and inspect IQ data from .wav files, check ADC utilization, and visualize the time-domain signal
- **spectrum.py** - Generate frequency spectrum plots and waterfall spectrograms to see what's happening in the frequency domain
- **demod.py** - Demodulate FM signals to recover the original audio using phase differentiation

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

Run any script with `-h` or `--help` for more information.

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