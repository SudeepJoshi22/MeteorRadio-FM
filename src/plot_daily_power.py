import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import glob
from datetime import datetime, timedelta

RADAR_DIR = os.path.expanduser('~/radar_data/')

def plot_daily_power_db(year, month, day, interactive_mode):
    date_str = f"{year}{month:02d}{day:02d}"
    search_pattern = os.path.join(RADAR_DIR, f"SMP_*_{date_str}_*.npz")
    npz_files = sorted(glob.glob(search_pattern))
    
    if not npz_files:
        print(f"No .npz files found for {date_str}")
        return
    
    print(f"Found {len(npz_files)} .npz file(s). Plotting power in dB vs time...\n")
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for file in npz_files:
        try:
            with np.load(file) as data:
                obs_time_str = data['obs_time'].item()
                freq_mhz = data['centre_freq'].item() / 1e6
                sample_rate = data['sample_rate'].item()
                samples = data['samples']
                
                if len(samples) == 0:
                    continue
                
                # Compute instantaneous power in dB
                power_linear = np.abs(samples)**2
                power_db = 10 * np.log10(power_linear + 1e-12)  # avoid log(0)
                
                # Time vector
                duration = len(samples) / sample_rate
                t_relative = np.linspace(0, duration, len(power_db))
                dt_start = datetime.strptime(obs_time_str, '%Y-%m-%d %H:%M:%S.%f')
                t_absolute = [dt_start + timedelta(seconds=s) for s in t_relative]
                
                label = f"{freq_mhz:.1f} MHz ({obs_time_str.split('.')[0]})"
                
                # Plot in dB, no offset, semi-transparent if many files
                alpha = 0.8 if len(npz_files) < 50 else 0.5
                ax.plot(t_absolute, power_db, label=label, linewidth=1.2, alpha=alpha)
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file)}: {e}")
            continue
    
    if ax.has_data():
        ax.set_title(f'IQ Power (dB) vs. Time for {year}-{month:02d}-{day:02d}')
        ax.set_xlabel('UTC Time')
        ax.set_ylabel('Power (dB, arbitrary reference)')
        ax.grid(True, alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Optional: show legend only if not too many files
        if len(npz_files) <= 20:
            ax.legend(fontsize='small', ncol=2)
        
        if interactive_mode:
            plt.show()
        else:
            output = os.path.join(RADAR_DIR, f'daily_power_db_{year}{month:02d}{day:02d}.png')
            plt.savefig(output, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output}")
    else:
        print("No valid data to plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot raw IQ power in dB vs time from .npz files (no offset).")
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    parser.add_argument("-i", "--interactive", action='store_true', help="Show interactive plot")
    args = parser.parse_args()
    
    plot_daily_power_db(args.year, args.month, args.day, args.interactive)
