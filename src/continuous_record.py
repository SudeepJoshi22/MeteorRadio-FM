import asyncio
import numpy as np
import datetime
import os
import signal
import scipy.signal as scipy_signal
from rtlsdr import RtlSdr
import argparse

# --- Configuration ---
DATA_DIR = os.path.expanduser('~/radar_data/Continuous/')
SAMPLE_RATE = 300000          # Raw sample rate from SDR
DECIMATION = 8                # Decimation factor
FREQUENCY_OFFSET = -2000       # Hz offset if needed

# Create directory
os.makedirs(DATA_DIR, exist_ok=True)

# Global for continuous saving
continuous_samples = []
current_date = None
sdr = None

def save_continuous_data():
    global continuous_samples, current_date
    if not continuous_samples:
        return

    # Approximate start time from buffer size
    approx_duration = len(continuous_samples) * (1024 / SAMPLE_RATE)  # each read is 1024 samples
    start_time = datetime.datetime.now() - datetime.timedelta(seconds=approx_duration)
    date_str = start_time.strftime('%Y%m%d')
    filename = os.path.join(DATA_DIR, f'SMP_{int(centre_freq)}_{date_str}_CONTINUOUS.npz')

    full_array = np.concatenate(continuous_samples)

    np.savez_compressed(
        filename,
        obs_time=str(start_time),
        centre_freq=centre_freq,
        sample_rate=SAMPLE_RATE / DECIMATION,
        samples=full_array.astype('complex64')
    )

    print(f"Saving continuous data: {filename} ({len(full_array)} samples)")
    continuous_samples.clear()
    current_date = start_time.date()

def signal_handler(signum, frame):
    print("\nShutting down gracefully...")
    save_continuous_data()
    if sdr:
        sdr.close()
    exit(0)

async def continuous_stream():
    global sdr, continuous_samples, current_date

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = centre_freq + FREQUENCY_OFFSET
    sdr.gain = sdr_gain

    print(f"Starting continuous recording on {centre_freq / 1e6:.3f} MHz")
    print(f"Gain: {sdr_gain}")
    print(f"Sample rate: {SAMPLE_RATE} Hz → decimated to {SAMPLE_RATE/DECIMATION:.1f} Hz")
    print(f"Saving daily files to: {DATA_DIR}")
    print("Press Ctrl+C to stop and save final file.\n")

    async for samples in sdr.stream():
        now = datetime.datetime.now()

        # Verbose per-block print (like original)
        if verbose:
            print(f"{now.strftime('%Y-%m-%d %H:%M:%S')} | Received block: {len(samples)} samples")

        # Decimate and append
        decimated = scipy_signal.decimate(samples, DECIMATION)
        continuous_samples.append(decimated)

        # Save on date rollover
        if current_date != now.date():
            save_continuous_data()
            current_date = now.date()

    await sdr.stop()
    sdr.close()
    save_continuous_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple continuous RTL-SDR recorder (daily .npz files)")
    parser.add_argument("-f", "--frequency", type=float, required=True, help="Center frequency in Hz (e.g. 107.1e6)")
    parser.add_argument("-g", "--gain", type=str, default='auto', help="Gain: number or 'auto'")
    parser.add_argument("-d", "--decimation", type=int, default=DECIMATION, help="Decimation factor")
    parser.add_argument("-v", "--verbose", action='store_true', help="Verbose block printing")
    args = parser.parse_args()

    centre_freq = args.frequency
    sdr_gain = args.gain
    DECIMATION = args.decimation
    verbose = args.verbose

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(continuous_stream())
    except KeyboardInterrupt:
        signal_handler(None, None)
