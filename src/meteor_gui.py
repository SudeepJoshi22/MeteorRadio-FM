import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import glob
from datetime import datetime, timedelta

RADAR_DIR = os.path.expanduser('~/radar_data/')

def plot_hourly_power_db(year, month, day, hour, interactive_mode):
    date_str = f"{year}{month:02d}{day:02d}"
    search_pattern = os.path.join(RADAR_DIR, f"SMP_*_{date_str}_*.npz")
    npz_files = sorted(glob.glob(search_pattern))
    
    if not npz_files:
        print(f"No .npz files found for {date_str}")
        return
    
    # Group by hour
    files_by_hour = {}
    for file in npz_files:
        try:
            with np.load(file) as data:
                obs_time_str = data['obs_time'].item()
                dt = datetime.strptime(obs_time_str, '%Y-%m-%d %H:%M:%S.%f')
                h = dt.hour
                if hour != -1 and h != hour:
                    continue
                if h not in files_by_hour:
                    files_by_hour[h] = []
                files_by_hour[h].append(file)
        except Exception:
            continue
    
    hours_to_plot = sorted(files_by_hour.keys()) if hour == -1 else [hour] if hour in files_by_hour else []
    
    if not hours_to_plot:
        print("No data for the specified hour.")
        return
    
    for h in hours_to_plot:
        fig, ax = plt.subplots(figsize=(15, 8))
        files = files_by_hour[h]
        print(f"Hour {h:02d}: Combining {len(files)} files...")
        
        all_t = []
        all_power_db = []
        
        for file in files:
            try:
                with np.load(file) as data:
                    obs_time_str = data['obs_time'].item()
                    sample_rate = data['sample_rate'].item()
                    samples = data['samples']
                    
                    if len(samples) == 0:
                        continue
                    
                    # Remove DC bias
                    i_mean = np.mean(samples.real)
                    q_mean = np.mean(samples.imag)
                    samples -= (i_mean + 1j * q_mean)
                    
                    # Power in dB
                    power_linear = np.abs(samples)**2
                    power_db = 10 * np.log10(power_linear + 1e-12)
                    
                    # Time vector
                    duration = len(samples) / sample_rate
                    t_relative = np.linspace(0, duration, len(power_db))
                    dt_start = datetime.strptime(obs_time_str, '%Y-%m-%d %H:%M:%S.%f')
                    t_absolute = [dt_start + timedelta(seconds=s) for s in t_relative]
                    
                    # Append to combined data
                    all_t.extend(t_absolute)
                    all_power_db.extend(power_db)
                    
            except Exception as e:
                print(f"Error in {os.path.basename(file)}: {e}")
        
        if all_t:
            ax.plot(all_t, all_power_db, linewidth=1, alpha=0.8)
            
            ax.set_title(f'Combined IQ Power (dB) vs. Time - {year}-{month:02d}-{day:02d} Hour {h:02d} UTC')
            ax.set_xlabel('UTC Time')
            ax.set_ylabel('Power (dB, arbitrary)')
            ax.grid(True, alpha=0.4)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
            if interactive_mode:
                plt.show()
            else:
                output = os.path.join(RADAR_DIR, f'power_db_combined_h{h:02d}_{year}{month:02d}{day:02d}.png')
                plt.savefig(output, dpi=150)
                print(f"Plot for hour {h:02d} saved to {output}")
        else:
            print(f"No data for hour {h:02d}")
        
        plt.close(fig)  # Free memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined power in dB vs time from .npz files, per hour.")
    parser.add_argument("-y", "--year", type=int, required=True)
    parser.add_argument("-m", "--month", type=int, required=True)
    parser.add_argument("-d", "--day", type=int, required=True)
    parser.add_argument("-hr", "--hour", type=int, default=-1, help="Specific hour (0-23) or -1 for all")
    parser.add_argument("-i", "--interactive", action='store_true')
    args = parser.parse_args()
    
    plot_hourly_power_db(args.year, args.month, args.day, args.hour, args.interactive)
