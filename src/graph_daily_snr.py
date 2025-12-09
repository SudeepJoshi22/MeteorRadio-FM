import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import glob
import re
from datetime import datetime

LOG_DIR = os.path.expanduser('~/radar_data/Logs/')

def plot_daily_snr(year, month, day, interactive_mode):
    """
    Finds all logs for a specific day, plots the peak SNR vs. UTC time for each,
    and displays the metadata from the filename in the legend.
    """
    # Find all log files for the specified day
    date_str = f"{year}-{month:02d}-{day:02d}"
    search_pattern = os.path.join(LOG_DIR, f"{date_str}_*.csv")
    log_files = glob.glob(search_pattern)

    if not log_files:
        print(f"No log files found for {date_str} with pattern {search_pattern}")
        return

    print(f"Found {len(log_files)} log file(s) for {date_str}.")

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(15, 7))

    for log_file in log_files:
        print(f"Processing {os.path.basename(log_file)}...")
        
        # --- Parse metadata from filename ---
        freq, gain, decimation = "N/A", "N/A", "N/A"
        match = re.search(r'_f(\d+\.?\d*)_g(auto|-?\d+)_d(\d+)\.csv', os.path.basename(log_file))
        if match:
            freq = match.group(1)
            gain = match.group(2)
            decimation = match.group(3)
        
        legend_label = f"f={freq}MHz, g={gain}, d={decimation}"

        # --- Read and process data ---
        try:
            df = pd.read_csv(log_file)
            if df.empty:
                print(f"Log file {os.path.basename(log_file)} is empty. Skipping.")
                continue
            
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            
            # --- Plot data ---
            ax.plot(df['datetime'], df['snratio'], marker='.', linestyle='-', markersize=4, label=legend_label)

        except Exception as e:
            print(f"Error processing file {os.path.basename(log_file)}: {e}")
            continue

    # --- Formatting the plot ---
    if not ax.has_data():
        print("No data was plotted. Aborting.")
        return

    ax.set_title(f'Peak SNR vs. Time for {date_str}')
    ax.set_xlabel('UTC Time')
    ax.set_ylabel('Peak SNR (dB)')
    ax.grid(True)
    ax.legend()

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    fig.autofmt_xdate()

    # --- Show or Save the plot ---
    if interactive_mode:
        print("\nDisplaying interactive plot. Close the window to exit.")
        plt.show()
    else:
        output_filename = os.path.join(LOG_DIR, f'daily_snr_plot_{date_str}.png')
        plt.savefig(output_filename)
        print(f"\nPlot saved successfully to {output_filename}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot the peak SNR vs. UTC time for a specific day from metadata-named log files.")
    ap.add_argument("-y", "--year", type=int, required=True, help="Year to plot (e.g., 2025)")
    ap.add_argument("-m", "--month", type=int, required=True, help="Month to plot (e.g., 12)")
    ap.add_argument("-d", "--day", type=int, required=True, help="Day to plot (e.g., 10)")
    ap.add_argument("-i", "--interactive", action='store_true', help="Display an interactive plot instead of saving to a file.")
    args = vars(ap.parse_args())

    plot_daily_snr(args['year'], args['month'], args['day'], args['interactive'])