import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import glob
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.dates as mdates

RADAR_DIR = os.path.expanduser('~/radar_data/')

def load_files_by_hour(year, month, day):
    date_str = f"{year}{month:02d}{day:02d}"
    pattern = os.path.join(RADAR_DIR, f"SMP_*_{date_str}_*.npz")
    files = sorted(glob.glob(pattern))
    by_hour = {}
    for f in files:
        try:
            with np.load(f) as data:
                t_str = data['obs_time'].item()
                t = datetime.strptime(t_str, '%Y-%m-%d %H:%M:%S.%f')
                h = t.hour
                by_hour.setdefault(h, []).append(f)
        except Exception:
            continue
    return by_hour

# --- Main GUI ---
class MeteorGUI:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
        self.files_by_hour = load_files_by_hour(year, month, day)
        self.hours = sorted(self.files_by_hour.keys())
        
        if not self.hours:
            print("No data found for this date!")
            return
        
        print(f"Found data for hours: {self.hours}")
        
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.buttons = []
        button_width = 0.07
        button_height = 0.07
        start_x = 0.05
        
        for i, h in enumerate(self.hours):
            ax_button = plt.axes([start_x + i * (button_width + 0.01), 0.05, button_width, button_height])
            btn = Button(ax_button, f'{h:02d}h', hovercolor='lightgray')
            btn.on_clicked(lambda event, hh=h: self.plot_hour(hh))
            self.buttons.append(btn)
        
        self.plot_hour(self.hours[0])  # initial plot
        plt.show()
    
    def plot_hour(self, hour):
        self.ax.clear()
        files = self.files_by_hour[hour]
        print(f"Plotting hour {hour:02d} — {len(files)} captures")
        
        for file in files:
            try:
                with np.load(file) as data:
                    samples = data['samples']
                    if len(samples) == 0:
                        continue
                    
                    freq = data['centre_freq'].item() / 1e6
                    rate = data['sample_rate'].item()
                    start_time = datetime.strptime(data['obs_time'].item(), '%Y-%m-%d %H:%M:%S.%f')
                    
                    # Downsample to ~10k points max
                    ds = max(1, len(samples) // 10000)
                    power_db = 10 * np.log10(np.abs(samples[::ds])**2 + 1e-12)
                    
                    duration = len(samples) / rate
                    t_rel = np.linspace(0, duration, len(power_db))
                    t_abs = [start_time + timedelta(seconds=s) for s in t_rel]
                    
                    self.ax.plot(t_abs, power_db, linewidth=1, alpha=0.7)
                    
            except Exception as e:
                print(f"Error loading {os.path.basename(file)}: {e}")
        
        self.ax.set_title(f'IQ Power (dB) vs Time — {self.year}-{self.month:02d}-{self.day:02d} Hour {hour:02d} UTC')
        self.ax.set_xlabel('UTC Time')
        self.ax.set_ylabel('Power (dB, arbitrary)')
        self.ax.grid(True, alpha=0.4)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        self.fig.autofmt_xdate()
        self.fig.canvas.draw()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone GUI for meteor scatter power plots (hour buttons)")
    parser.add_argument("-y", "--year", type=int, required=True, help="Year (e.g. 2025)")
    parser.add_argument("-m", "--month", type=int, required=True, help="Month (e.g. 12)")
    parser.add_argument("-d", "--day", type=int, required=True, help="Day (e.g. 29)")
    args = parser.parse_args()
    
    gui = MeteorGUI(args.year, args.month, args.day)
