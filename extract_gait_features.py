import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# Load the first si subsets dataset for retrive the gait features and combined it with demographics data
df = pd.read_csv("dataset_clean.csv")
df.columns = df.columns.astype(str).str.strip().str.lower()

# Rename columns based on known structure
df.rename(columns={                                                                                                                             
    '0': 'time',
    '17': 'total_left',
    '18': 'total_right'
}, inplace=True)

# Drop unused column if needed here the siurce_file is not needed anymore so we drop it
if 'source_file' in df.columns:
    df.drop(columns=['source_file'], inplace=True)

# Extract relevant columns for gait features extraction
time = df['time'].values
vgrf_left = df['total_left'].values
vgrf_right = df['total_right'].values

# Calculate sampling rate
sampling_rate = 1 / np.mean(np.diff(time))

# Detect peaks
min_step_interval_sec = 0.5
distance_threshold = int(min_step_interval_sec * sampling_rate)
peaks_left, _ = find_peaks(vgrf_left, height=np.max(vgrf_left) * 0.2, distance=distance_threshold)
peaks_right, _ = find_peaks(vgrf_right, height=np.max(vgrf_right) * 0.2, distance=distance_threshold)

# Gait parameters like cadence, stride regularity etc..
total_steps = len(peaks_left) + len(peaks_right)
total_time_min = (time[-1] - time[0]) / 60
cadence = total_steps / total_time_min

step_intervals_left = np.diff(time[peaks_left])
step_intervals_right = np.diff(time[peaks_right])

# Compute regularity via autocorrelation
def compute_regularity(signal):
    signal = signal - np.mean(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    return autocorr[1] / autocorr[0] if autocorr[0] != 0 else 0

step_regularity = compute_regularity(vgrf_left)
stride_regularity = compute_regularity(vgrf_right)

# Compute symmetry using Pearson correlation
min_len = min(len(vgrf_left), len(vgrf_right))
sym1 = pearsonr(vgrf_left[:min_len], vgrf_right[:min_len])[0]
sym2 = pearsonr(vgrf_left[:min_len], -vgrf_right[:min_len])[0]
step_symmetry = sym1 if abs(sym1) > abs(sym2) else sym2
stride_symmetry = step_symmetry

# Stride length estimation
avg_stride_time = (np.mean(step_intervals_left) + np.mean(step_intervals_right)) / 2
simulated_speed = 1.2  # meters per second
stride_length = simulated_speed * avg_stride_time

# Extract Output features
features = {
    "cadence": round(cadence, 2),
    "stride_length": round(stride_length, 2),
    "step_regularity": round(step_regularity, 3),
    "stride_regularity": round(stride_regularity, 3),
    "step_symmetry": round(step_symmetry, 3),
    "stride_symmetry": round(stride_symmetry, 3),
    "step_count": total_steps
}

print("\nExtracted Gait Features:")
for k, v in features.items():
    print(f"{k}: {v}")
