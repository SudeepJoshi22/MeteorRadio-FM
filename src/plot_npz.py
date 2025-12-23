import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python3 plot_npz.py <filename.npz>")
    sys.exit(1)

file = sys.argv[1]

with np.load(file) as data:
    samples = data['samples']
    rate = data['sample_rate']
    freq = data['centre_freq'] / 1e6
    
    power = np.abs(samples)**2
    t = np.arange(len(power)) / rate
    
    plt.figure(figsize=(12,4))
    plt.plot(t, 10*np.log10(power + 1e-10))
    plt.title(f'{freq:.1f} MHz - {data["obs_time"].item()}')
    plt.xlabel('Time (s)')
    plt.ylabel('Power (dB)')
    plt.grid()
    plt.show()
