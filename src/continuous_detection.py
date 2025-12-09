import glob
import time
from rtlsdr import *
import asyncio
import numpy as np
import datetime
import re
import scipy.signal as scipy_signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
import scipy.io.wavfile as wav
from collections import deque
from queue import Queue
import os
import syslog
import threading
import signal
import argparse
from multiprocessing import Process, Queue as mpQueue
from waterfall import Waterfall

# --- Configuration ---
DATA_DIR = os.path.expanduser('~/radar_data/')
CAPTURES_DIR = DATA_DIR + 'Captures/'
ARCHIVE_DIR = DATA_DIR + 'Archive/'
LOG_DIR = DATA_DIR + 'Logs/'
CONFIG_FILE = os.path.expanduser('~/.radar_config')
DISK_SPACE_TO_LEAVE = 1e9  # 1GB

SAMPLES_LENGTH = 24
SAMPLES_BEFORE_TRIGGER = 6

SAMPLE_RATE = 300000
SDR_GAIN = 50
DECIMATION = 8

FREQUENCY_OFFSET = -2000
DETECTION_FREQUENCY_BAND = [-120, 120]
NOISE_CALCULATION_BAND = [-500, 500]
OVERLAP = 0.75
ANALYSIS_OVERLAP = 0.5
COMPRESSION_FREQUENCY_BAND = 1000
AUDIO_FREQUENCY_BANDPASS = [1500, 3000]
NUM_FFT = 2**15
HOP = int(NUM_FFT * (1 - ANALYSIS_OVERLAP))

TRIGGERS_REQUIRED = 1
MAX_MEDIAN_NOISE_RATIO = 3

# --- Core Classes and Functions ---

def signalHandler(signum, frame):
    if signum == signal.SIGUSR1:
        syslog.syslog(syslog.LOG_DEBUG, "SIGUSR1 caught")
        sample_analyser.save_samples()
    else:
        os._exit(0)

def make_directories():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

class TimedSample:
    def __init__(self, sample, sample_time):
        self.sample = sample
        self.sample_time = sample_time

class DiskSpaceChecker(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while True:
            try:
                statvfs = os.statvfs(os.path.expanduser('~'))
                space_left = statvfs.f_frsize * statvfs.f_bavail
                if space_left > DISK_SPACE_TO_LEAVE:
                    time.sleep(1800)
                    continue

                radio_files = glob.glob(os.path.join(DATA_DIR, '**/SMP*.npz'), recursive=True)
                radio_files += glob.glob(os.path.join(DATA_DIR, '**/SPG*.npz'), recursive=True)
                radio_files += glob.glob(os.path.join(DATA_DIR, '**/AUD*.raw'), recursive=True)
                radio_files = sorted(radio_files, key=os.path.getmtime)

                while True:
                    statvfs = os.statvfs(os.path.expanduser('~'))
                    space_left = statvfs.f_frsize * statvfs.f_bavail
                    if space_left > DISK_SPACE_TO_LEAVE or len(radio_files) == 0:
                        break
                    os.remove(radio_files.pop(0))
            except Exception as e:
                syslog.syslog(syslog.LOG_DEBUG, f"DiskSpaceChecker error: {e}")
            time.sleep(1800)

class RMBLogger:
    def __init__(self):
        self.id, self.Long, self.Lat, self.Alt = "", 0.0, 0.0, 0.0
        self.Ver, self.Tz = "RMOB", 0
        self.get_config()

    def get_config(self):
        try:
            with open(CONFIG_FILE) as fp:
                for line in fp:
                    parts = re.split("[: \n]+", line)
                    if parts[0] == 'stationID': self.id = parts[1]
                    if parts[0] == 'latitude': self.Lat = float(parts[1])
                    if parts[0] == 'longitude': self.Long = float(parts[1])
                    if parts[0] == 'elevation': self.Alt = float(parts[1])
        except Exception as e:
            syslog.syslog(syslog.LOG_DEBUG, f"RMBLogger config error: {e}")

    def log_data(self, obs_time, Bri, Dur, freq):
        filename = f"R{obs_time.strftime('%Y%m%d_')}{self.id}.csv"
        filepath = os.path.join(LOG_DIR, filename)
        try:
            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    f.write("Ver,Y,M,D,h,m,s,Bri,Dur,freq,ID,Long,Lat,Alt,Tz\n")
            
            with open(filepath, "a") as f:
                rmb_string = f'{self.Ver},{obs_time.strftime("%Y,%m,%d,%H,%M,%S.%f")[:-3]},{Bri:.2f},{Dur:.2f},{freq:.2f},{self.id},{self.Long:.5f},{self.Lat:.5f},{self.Alt:.1f},{self.Tz}\n'
                f.write(rmb_string)
        except Exception as e:
            syslog.syslog(syslog.LOG_DEBUG, f"RMBLogger write error: {e}")

class ParameterCsvLogger:
    def __init__(self, frequency, gain, decimation):
        self.id, self.Lat, self.Long, self.foff = "", 0.0, 0.0, 0.0
        self.tx_source, self.time_sync = "", ""
        self.get_config()
        
        self.freq_mhz = frequency / 1e6
        self.gain = gain
        self.decimation = decimation

    def get_config(self):
        try:
            with open(CONFIG_FILE) as fp:
                for line in fp:
                    parts = re.split("[: \n]+", line)
                    if parts[0] == 'ID_NUM': self.id = parts[1]
                    if parts[0] == 'latitude': self.Lat = float(parts[1])
                    if parts[0] == 'longitude': self.Long = float(parts[1])
                    if parts[0] == 'foff': self.foff = float(parts[1])
                    if parts[0] == 'TxSource': self.tx_source = parts[1]
                    if parts[0] == 'TimeSync': self.time_sync = parts[1]
        except Exception as e:
            syslog.syslog(syslog.LOG_DEBUG, f"ParameterCsvLogger config error: {e}")

    def get_filename(self, obs_time):
        date_str = obs_time.strftime('%Y-%m-%d')
        return f"{date_str}_f{self.freq_mhz:.1f}_g{self.gain}_d{self.decimation}.csv"

    def log_data(self, obs_time, centre_freq, frequency, signal, noise, duration, max_snr):
        try:
            filename = self.get_filename(obs_time)
            filepath = os.path.join(LOG_DIR, filename)

            if not os.path.exists(filepath):
                with open(filepath, "w") as f:
                    f.write("user_ID,date,time,signal,noise,frequency,durationc,durations,lat,long,source,timesync,snratio,doppler_estimate\n")

            date = obs_time.strftime('%Y-%m-%d')
            time = obs_time.strftime('%H:%M:%S.%f')
            doppler_estimate = int(float(frequency) - float(centre_freq) - self.foff)
            offset_frequency = int(2000 + (float(frequency)) - float(centre_freq) - self.foff)
            output_line = f"{self.id},{date},{time},{signal:.3f},{noise:.3f},{offset_frequency},0,{duration:.2f},{self.Lat:.2f},{self.Long:.2f},{self.tx_source},{self.time_sync},{max_snr:.2f},{doppler_estimate}\n"
            
            with open(filepath, "a") as f:
                f.write(output_line)
        except Exception as e:
            syslog.syslog(syslog.LOG_DEBUG, f"ParameterCsvLogger write error: {e}")

class CaptureStatistics:
    def __init__(self, Pxx, f, bins, obs_time, snr_threshold, centre_freq):
        self.Pxx, self.f, self.bins = Pxx, f, bins
        self.obs_time, self.snr_threshold, self.centre_freq = obs_time, snr_threshold, centre_freq
        self.noise_calculation_band = np.where((f * 1e6 > (self.centre_freq + NOISE_CALCULATION_BAND[0])) & (f * 1e6 <= (self.centre_freq + NOISE_CALCULATION_BAND[1])))

    def calculate(self):
        if self.Pxx.size == 0:
            self.snr = 0
            self.log_mn = 0
            self.log_sigmax = 0
            self.peak_freq = 0
            self.detection_duration = 0
            self.detection_freq = 0
            self.detection_time = self.obs_time
            return

        nx = np.float16(self.Pxx[self.noise_calculation_band])
        self.raw_median = np.median(self.Pxx[0:13]) if self.Pxx.shape[1] > 13 else np.median(self.Pxx)
        self.log_mn = 10.0 * np.log10(self.raw_median)
        self.log_sigmax = 10.0 * np.log10(np.max(self.Pxx))
        self.peak_freq = self.f[np.argmax(np.max(self.Pxx, axis=1))]
        self.snr = self.log_sigmax - self.log_mn
        self.detection_duration = self.bins[1] - self.bins[0] if len(self.bins) > 1 else 0
        self.detection_freq = self.peak_freq
        self.detection_time = self.obs_time + datetime.timedelta(seconds=2)

class SampleAnalyser(threading.Thread):
    def __init__(self, centre_freq, sdr_gain, decimation_factor):
        threading.Thread.__init__(self)
        self.noise_deque = deque(maxlen=8)
        self.trigger_count = 0
        self.trigger_wait_counter = 0
        self.analysis_thread = None
        self.save_process1 = None
        self.save_process2 = None
        self.centre_freq = centre_freq
        self.decimation_factor = decimation_factor
        self.rmb_logger = RMBLogger()
        self.csv_logger = ParameterCsvLogger(centre_freq, sdr_gain, decimation_factor)
        self.captures_dir = DATA_DIR

    def run(self):
        global sdr
        psd_queue = mpQueue(maxsize=4)
        samples = sample_queue.get()
        self.sdr_freq = sdr.center_freq
        self.sdr_freq_mhz = sdr.center_freq / 1e6
        self.sdr_sample_rate = sdr.sample_rate
        self.sample_time = len(samples) / self.sdr_sample_rate
        self.decimated_sample_rate = self.sdr_sample_rate / self.decimation_factor

        window = hamming(NUM_FFT, sym=True)
        sft = ShortTimeFFT(window, hop=HOP, fs=self.sdr_sample_rate, mfft=NUM_FFT, fft_mode='centered')
        Pxx = sft.spectrogram(samples)
        f = sft.f / 1e6 + self.sdr_freq_mhz
        self.noise_calculation_band = np.where((f * 1e6 > (self.centre_freq + NOISE_CALCULATION_BAND[0])) & (f * 1e6 <= (self.centre_freq + NOISE_CALCULATION_BAND[1])))
        self.detection_band = np.where((f * 1e6 > (self.centre_freq +DETECTION_FREQUENCY_BAND[0])) & (f * 1e6 <= (self.centre_freq +DETECTION_FREQUENCY_BAND[1])))

        while True:
            samples = sample_queue.get()
            if not psd_queue.full():
                self.analysis_thread = Process(target=self.analyse_psd, args=(np.copy(samples), psd_queue))
                self.analysis_thread.start()
            while not psd_queue.empty():
                psd_results = psd_queue.get()
                self.check_trigger(psd_results)

    def check_trigger(self, psd_results):
        mn, sigmedian, sigmax, peak_freq, ratio_median = psd_results
        stats = f' Mean:{mn:8.4f}  Median:{sigmedian:8.4f}  Max:{sigmax:10.4f}  PeakF:{peak_freq:12.6f}  SNR:{sigmax/sigmedian:10.2f}'
        if verbose: print(datetime.datetime.now(), stats)
        
        trigger = True
        if trigger:
            if self.trigger_count == 0:
                syslog.syslog(syslog.LOG_DEBUG, f"Continuous recording trigger at {datetime.datetime.now()} {stats}")
            self.trigger_count += 1
        else:
            self.noise_deque.append(mn)

        if self.trigger_count >=TRIGGERS_REQUIRED:
            self.trigger_wait_counter += 1
            if self.trigger_wait_counter >= SAMPLES_LENGTH - SAMPLES_BEFORE_TRIGGER:
                self.save_samples()
                self.trigger_count = 0
                self.trigger_wait_counter = 0
        elif not trigger:
            self.trigger_count = 0

    def save_samples(self):
        timed_sample_snapshot = timed_sample_deque.copy()
        obs_time = timed_sample_snapshot[0].sample_time - datetime.timedelta(seconds=self.sample_time)
        all_samples = np.asarray([ts.sample for ts in timed_sample_deque]).flatten()
        
        if capturetodated:
            self.captures_dir = os.path.join(CAPTURES_DIR, obs_time.strftime('%Y%m%d'))
            os.makedirs(self.captures_dir, exist_ok=True)

        if save_raw_samples:
            if self.save_process1 is None or not self.save_process1.is_alive():
                self.save_process1 = Process(target=self.save_raw_sample_data, args=(all_samples, self.sdr_freq, self.centre_freq, self.sdr_sample_rate, obs_time))
                self.save_process1.start()
            elif self.save_process2 is None or not self.save_process2.is_alive():
                self.save_process2 = Process(target=self.save_raw_sample_data, args=(all_samples, self.sdr_freq, self.centre_freq, self.sdr_sample_rate, obs_time))
                self.save_process2.start()

    def save_raw_sample_data(self, raw_samples, sdr_centre_freq, centre_freq, sample_rate, obs_time):
        decimated_samples = scipy_signal.decimate(raw_samples, self.decimation_factor)
        sample_filename = os.path.join(self.captures_dir, f'SMP_{int(centre_freq)}_{obs_time.strftime("%Y%m%d_%H%M%S_%f")}.npz')
        syslog.syslog(syslog.LOG_DEBUG, f"Saving {sample_filename}")
        np.savez(sample_filename, obs_time=str(obs_time), centre_freq=centre_freq, sample_rate=self.decimated_sample_rate, samples=np.array(decimated_samples).astype("complex64"))
        print(f"Saving {sample_filename}")
        self.log_capture_stats(raw_samples, sdr_centre_freq, sample_rate, obs_time)

    def log_capture_stats(self, raw_samples, sdr_centre_freq, sample_rate, obs_time):
        window = hamming(NUM_FFT, sym=True)
        sft = ShortTimeFFT(window, hop=HOP, fs=sample_rate, mfft=NUM_FFT, fft_mode='centered')
        Pxx = sft.spectrogram(raw_samples)
        
        sdr_freq_mhz = sdr_centre_freq / 1e6
        f = sft.f / 1e6 + sdr_freq_mhz
        
        bins = np.arange(Pxx.shape[1]) * (HOP / sample_rate)
        
        freq_slice = np.where((f >= (self.centre_freq - COMPRESSION_FREQUENCY_BAND) / 1e6) & (f <= (self.centre_freq + COMPRESSION_FREQUENCY_BAND) / 1e6))
        
        f_sliced = f[freq_slice]
        Pxx_sliced = Pxx[freq_slice]

        if f_sliced.size == 0 or Pxx_sliced.size == 0:
            syslog.syslog(syslog.LOG_DEBUG, "Error: No data in frequency slice for statistics calculation.")
            return

        stats = CaptureStatistics(Pxx_sliced, f_sliced, bins, obs_time, snr_threshold, self.centre_freq)
        stats.calculate()
        
        if np.isnan(stats.snr):
             syslog.syslog(syslog.LOG_DEBUG, "Error: SNR calculation resulted in NaN.")
             return

        self.rmb_logger.log_data(stats.detection_time, stats.snr, stats.detection_duration, (stats.detection_freq * 1e6) - self.centre_freq)
        self.csv_logger.log_data(stats.detection_time, self.centre_freq, stats.detection_freq * 1e6, stats.log_sigmax, stats.log_mn, stats.detection_duration, stats.snr)

    def analyse_psd(self, samples, psd_queue):
        window = hamming(NUM_FFT, sym=True)
        sft = ShortTimeFFT(window, hop=HOP, fs=self.sdr_sample_rate, mfft=NUM_FFT, fft_mode='centered')
        Pxx = sft.spectrogram(samples)
        f = sft.f / 1e6 + self.sdr_freq_mhz
        
        nx = np.float16(Pxx[self.noise_calculation_band])
        mn = np.mean(nx) if nx.size > 0 else 0
        
        sigmedian = 0
        if nx.size > 0:
            max_index_flat = np.argmax(nx)
            _, col_idx = np.unravel_index(max_index_flat, nx.shape)
            sigmedian = np.median(nx[:, col_idx])
        
        x = np.float16(Pxx[self.detection_band])
        sigmax = np.max(x) if x.size > 0 else 0
        peak_freq = f[self.detection_band][np.argmax(np.max(x, axis=1))] if x.size > 0 else 0
        
        time_medians = np.median(Pxx, axis=0)
        ratio_median = np.max(time_medians) / np.median(time_medians) if time_medians.size > 0 else 0
        psd_queue.put((mn, sigmedian, sigmax, peak_freq, ratio_median))

async def streaming():
    global sdr
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = centre_freq + FREQUENCY_OFFSET
    if sdr_gain == 'auto':
        sdr.gain = 'auto'
    else:
        sdr.gain = float(sdr_gain)
    async for samples in sdr.stream():
        timed_sample_deque.append(TimedSample(samples, datetime.datetime.now()))
        sample_queue.put(samples)
        if display_waterfall:
            try:
                if waterfall_queue.full(): waterfall_queue.get_nowait()
                waterfall_queue.put_nowait(samples)
            except: pass
    await sdr.stop()
    sdr.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signalHandler)
    signal.signal(signal.SIGTERM, signalHandler)
    signal.signal(signal.SIGUSR1, signalHandler)

    ap = argparse.ArgumentParser(description="Continuous radio data recorder for meteor detection.")
    ap.add_argument("-f", "--frequency", type=float, default=143.05e6, help="Centre frequency in Hz.")
    ap.add_argument("-g", "--gain", type=str, default=str(SDR_GAIN), help="SDR tuner gain (0-50, or 'auto').")
    ap.add_argument("-s", "--snr_threshold", type=float, default=45, help="SNR threshold for statistics (not for triggering).")
    ap.add_argument("-r", "--raw", action='store_true', default=True, help="Store raw sample data (default).")
    ap.add_argument("--fft", action='store_true', help="Store data as FFT (not fully supported in this script).")
    ap.add_argument("-a", "--audio", action='store_true', help="Enable saving of audio wav file (not fully supported in this script).")
    ap.add_argument("-d", "--decimation", type=int, default=DECIMATION, help=f"Decimation factor (default: {DECIMATION}).")
    ap.add_argument("-c", "--capturetodated", action='store_true', help="Store captures to dated directories.")
    ap.add_argument("-w", "--waterfall", action='store_true', help="Display waterfall graph.")
    ap.add_argument("-v", "--verbose", action='store_true', help="Verbose output.")
    args = vars(ap.parse_args())

    centre_freq = args['frequency']
    sdr_gain = args['gain']
    snr_threshold = args['snr_threshold']
    save_raw_samples = args['raw']
    DECIMATION = args['decimation']
    capturetodated = args['capturetodated']
    display_waterfall = args['waterfall']
    verbose = args['verbose']

    make_directories()

    timed_sample_deque = deque(maxlen=SAMPLES_LENGTH)
    sample_queue = Queue(maxsize=10)
    waterfall_queue = mpQueue(maxsize=1)
    
    sdr = RtlSdr()
    
    sample_analyser = SampleAnalyser(centre_freq, sdr_gain, DECIMATION)
    sample_analyser.start()
    
    diskspacechecker = DiskSpaceChecker()
    diskspacechecker.start()
    
    if display_waterfall:
        p = Waterfall(centre_freq + FREQUENCY_OFFSET, SAMPLE_RATE, waterfall_queue)
        p.start()

    asyncio.run(streaming())
