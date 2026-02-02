import numpy as np
import pandas as pd
import scipy.io
import os
import matplotlib.pyplot as plt
import seaborn as sns


# --- Configuration ---
spike_matrix_csv_file = 'all_spike_times_matrix_comma_sep.csv'
# --- ---

# Loading the spike time matrix CSV file
try:
    # 1. Load the CSV, specifying the first column ('neuron') as the index
    df = pd.read_csv(spike_matrix_csv_file, index_col='neuron')
    print(f"Loaded '{spike_matrix_csv_file}' into DataFrame.")

    # Get dimensions (adjusting for index column)
    num_neurons = df.shape[0]
    num_repetitions = df.shape[1]

    # 2. Create an empty NumPy array that can hold Python objects (like lists or other arrays)
    # Note: Indices will be 0 to num_neurons-1 and 0 to num_repetitions-1
    spike_vectors_array = np.empty((num_neurons, num_repetitions), dtype=object)

    print("Converting string data to numerical vectors...")
    # 3. Iterate and convert each cell
    for i in range(num_neurons): # Neuron index (0 to 3)
        for j in range(num_repetitions): # Repetition index (0 to 49)
            # Get the string from the DataFrame (using neuron number i+1 and repetition number j+1 as labels)
            # Use .loc for label-based indexing. Column names are strings '1', '2', ...
            spike_string = df.loc[i + 1, str(j + 1)]

            # Check if the string is not empty (handles cases with no spikes)
            if isinstance(spike_string, str) and spike_string.strip():
                # Split the string by comma, convert each part to float, store as numpy array
                spike_vectors_array[i, j] = np.array([float(t) for t in spike_string.split(',')])
            else:
                # Store an empty numpy array if there were no spikes
                spike_vectors_array[i, j] = np.array([], dtype=float)

    print("Conversion complete.")
    print("-" * 30)

except FileNotFoundError:
    print(f"Error: The file '{spike_matrix_csv_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Loading the stimulus CSV file
stimulus_csv_file = 'stimulus.csv'
try:
    stimulus_df = pd.read_csv(stimulus_csv_file)
    print(f"Loaded '{stimulus_csv_file}' into DataFrame.")

    # Extract stimulus values as a NumPy array
    stimulus_values = stimulus_df['stimulus_db'].values
    print("Stimulus values extracted.")
    print("-" * 30)
except FileNotFoundError:
    print(f"Error: The file '{stimulus_csv_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


'''
----------- Plotting the Stimulus Averages -----------
params:
- window_size: Number of samples to average over (e.g., 100)
'''
window_size = 100

# Ensure we have at least window_size samples
if stimulus_values.size < window_size:
    raise ValueError(f"stimulus_values must contain at least {window_size} samples")

# Use only the largest multiple of window_size (trim any remainder)
n_blocks = stimulus_values.size // window_size
trimmed = stimulus_values[: n_blocks * window_size]

# Reshape and compute mean across each block -> gives n_blocks averages (expected 200)
averages = trimmed.reshape(n_blocks, window_size).mean(axis=1)

print(f"Computed {averages.size} averages (window={window_size}).")
# --- end new code ---

# Plotting the averaged stimulus values
sns.set(style="whitegrid")
plt.figure(figsize=(10, 4))
sns.lineplot(x=np.arange(len(averages)), y=averages, marker='o')
plt.title(f'Stimulus Averages (window={window_size}, n={len(averages)})')
plt.xlabel('Block index')
plt.ylabel('Average Stimulus (dB)')
plt.grid(True)
plt.tight_layout()
plt.show()

'''
----------- Completed Plotting Stimulus Average----------
'''

'''
----------- Question - 1----------
'''
n = len(stimulus_values)
results = {}

# Ensure data is a numpy array
stimulus_values = np.array(stimulus_values)

# 'full' mode computes the correlation at all possible lags,
# from -(n-1) to +(n-1).
full_corr = np.correlate(stimulus_values, stimulus_values, mode='full')

# The 'full_corr' array has length 2*n - 1.
# The center element (lag 0) is at index (n - 1).
# The value for any lag 'tau' is at index (n - 1) + tau.

lag_0_index = n - 1

for tau in range(-50, 51):
    index = lag_0_index + tau
    
    # Check if the requested lag 'tau' is within the
    # possible range [-(n-1), (n-1)].
    if 0 <= index < len(full_corr):
        results[tau] = full_corr[index]
    else:
        # If the lag is outside the possible range, there is
        # no overlap, so the autocorrelation is 0.
        results[tau] = 0.0

plt.figure(figsize=(10, 4))
sns.barplot(x=list(results.keys()), y=list(results.values()), color='blue')
plt.title('Stimulus Autocorrelation (lags -50 to 50)')
plt.xlabel('Lag (tau)')
plt.ylabel('Autocorrelation')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

'''
----------- Question - 2----------
'''

# --- Question 2: PSTH (4 x 20000) ---
print('Computing PSTH...')
num_bins = 20000
PSTH = np.zeros((num_neurons, num_bins), dtype=float)
for i in range(num_neurons):
    for j in range(num_repetitions):
        spikes_s = spike_vectors_array[i, j]
        if spikes_s.size:
            spikes_ms = spikes_s * 1000.0
            counts, _ = np.histogram(spikes_ms, bins=np.arange(0, num_bins + 1))
        else:
            counts = np.zeros(num_bins, dtype=int)
        PSTH[i, :] += counts * 1000.0 / num_repetitions

print('PSTH computed.')

# Plot PSTH for 4 neurons
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for i in range(num_neurons):
    axes[i].plot(PSTH[i, :])
    axes[i].set_xlabel('time (ms)')
    axes[i].set_ylabel('rate (spikes/sec)')
    axes[i].set_title(f'PSTH for neuron {i+1}')
plt.tight_layout()
plt.show()


"""
----------- Question 3: POISSON PROCESS ----------
Use TEST_Q3=True to limit bin sizes for quick testing.
"""
# Set TEST_Q3 True for quick runs (reduced bins), False for full analysis
TEST_Q3 = False
bin_sizes = [10, 20, 50, 100, 200, 500]
if TEST_Q3:
    bin_sizes = [50, 100]
for bsize in bin_sizes:
    print(f'Processing bin size {bsize} ms...')
    plt.figure()
    for n in range(num_neurons):
        means = []
        variances = []
        for j in range(num_repetitions):
            spikes_s = spike_vectors_array[n, j]
            if spikes_s.size:
                spikes_ms = spikes_s * 1000.0
                bins = np.arange(0, num_bins + bsize, bsize)
                counts, _ = np.histogram(spikes_ms, bins=bins)
            else:
                counts = np.zeros(int(np.ceil(num_bins / bsize)), dtype=int)
            M = counts.mean()
            V = counts.var()
            means.append(M)
            variances.append(V)

        # --- New: ISI distribution and CV per repetition for this neuron ---
        # Gather ISIs across all repetitions
        all_isis = []
        cvs = []
        for j in range(num_repetitions):
            spikes_s = spike_vectors_array[n, j]
            if spikes_s.size and spikes_s.size > 1:
                isis = np.diff(spikes_s) * 1000.0  # in ms
                all_isis.extend(isis.tolist())
                cvs.append(np.std(isis) / np.mean(isis) if np.mean(isis) != 0 else np.nan)
            else:
                cvs.append(np.nan)

        # Plot ISI histogram (aggregated across repetitions)
        plt.figure()
        plt.hist(all_isis, bins=50, color='gray')
        plt.title(f'ISI distribution (Neuron {n+1}, bin size {bsize}ms)')
        plt.xlabel('Inter-spike interval (ms)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

        # Plot CV per repetition
        plt.figure()
        plt.plot(np.arange(1, num_repetitions + 1), cvs, marker='o')
        plt.title(f'CV of ISI per repetition (Neuron {n+1}, bin size {bsize}ms)')
        plt.xlabel('Repetition')
        plt.ylabel('CV (std/mean)')
        plt.ylim([0, np.nanmax(cvs) * 1.1 if len([c for c in cvs if not np.isnan(c)])>0 else 1])
        plt.tight_layout()
        plt.show()

        max_v = max(max(means), max(variances))
        min_v = min(min(means), min(variances))
        straight_line = [0, max_v]
        plt.subplot(2, 2, n + 1)
        plt.scatter(variances, means)
        plt.plot(straight_line, straight_line, color='orange')
        plt.title(f'Mean vs Variance for bin size {bsize}ms - Neuron {n+1}')
        plt.xlabel('variance')
        plt.ylabel('mean')
        plt.xlim([min_v, max_v])
        plt.ylim([min_v, max_v])
    plt.tight_layout()
    plt.show()


"""
----------- Question 4: SPIKE TRIGGERED AVERAGE ----------
"""
print('Computing STA and whitened filters...')
sta = np.zeros((num_neurons, 100), dtype=float)
h = np.zeros((num_neurons, 100), dtype=float)

# Compute autocorrelation Rxx for lags 0..99 using unbiased autocov
stim = stimulus_values.astype(float)
stim_mean = stim.mean()
stim_zero = stim - stim_mean
full_corr = np.correlate(stim_zero, stim_zero, mode='full')
lag0 = stim_zero.size - 1
Rxx = full_corr[lag0:lag0 + 100] / stim_zero.size

from scipy.linalg import toeplitz
Css = toeplitz(Rxx)

for i in range(num_neurons):
    total_spike_count = 0
    for j in range(num_repetitions):
        spikes = spike_vectors_array[i, j]
        # consider spikes <= 15 seconds
        mask = spikes <= 15.0
        total_spike_count += np.count_nonzero(mask)
        for t_spike in spikes[mask]:
            int_timing = int(round(t_spike * 1000.0))
            # Python 0-based: want samples int_timing-100 .. int_timing-1
            start = max(int_timing - 100, 0)
            end = int_timing
            stim_values_seg = stim[start:end]
            partial_STA = np.concatenate([np.zeros(100 - len(stim_values_seg)), stim_values_seg])
            sta[i, :] += partial_STA

    if total_spike_count > 0:
        sta[i, :] = sta[i, :] / total_spike_count
    else:
        sta[i, :] = np.zeros(100)

    # plot STA (flipped to match MATLAB fliplr in plotting)
    plt.figure(9)
    plt.subplot(2, 2, i + 1)
    plt.plot(np.arange(0, 100), sta[i, ::-1])
    plt.xlabel('time (ms)')
    plt.ylabel('STA')
    plt.ylim([-0.5, 0.5])
    plt.title(f'h(t) without correction for neuron {i+1}')

    # whitening correction: solve Css * h_rev = sta'
    try:
        h_rev = np.linalg.solve(Css, sta[i, :])
    except np.linalg.LinAlgError:
        # fallback to pseudo-inverse
        h_rev = np.linalg.pinv(Css) @ sta[i, :]
    h[i, :] = h_rev[::-1]
    plt.figure(10)
    plt.subplot(2, 2, i + 1)
    plt.plot(np.arange(0, 100), h[i, :])
    plt.xlabel('time (ms)')
    plt.ylabel('h(t)')
    plt.ylim([-1, 1])
    plt.title(f'h(t) with correction for neuron {i+1}')

plt.show()


"""
----------- Part 5 / Question 5: Predictions and fitting nonlinearities ----------
"""
stimulus = stim
psth = PSTH / 1000.0

# linear predictions using first 15000 samples
pred = np.zeros((num_neurons, 15000 + 1000))
xpred = np.zeros((num_neurons, 15000))
for i in range(num_neurons):
    conv_res = np.convolve(stimulus[:15000], h[i, :])
    xpred[i, :15000] = conv_res[:15000]

# ground truth rates
y_rates = np.zeros((num_neurons, 15000))
for i in range(num_neurons):
    y_rates[i, :] = 1000.0 * psth[i, :15000]

# bin and average with bin_size 30
bin_size = 30
n_bins = int(np.ceil(15000 / bin_size))
xbins = [[] for _ in range(num_neurons)]
ybins = [[] for _ in range(num_neurons)]
for b in range(n_bins):
    s = b * bin_size
    e = min((b + 1) * bin_size, 15000)
    for i in range(num_neurons):
        xbins[i].append(xpred[i, s:e].mean())
        ybins[i].append(y_rates[i, s:e].mean())

for i in range(num_neurons):
    xbins[i] = np.array(xbins[i])
    ybins[i] = np.array(ybins[i])

# Fit sigmoid a/(1+exp(-b*(x-c))) using scipy.optimize.curve_fit
from scipy.optimize import curve_fit
from scipy.special import expit

def sigmoid_unstable(x, a, b, c):
    return a / (1 + np.exp(-b * (x - c)))

def sigmoid(x, a, b, c):
    # numerically stable via expit
    return a * expit(b * (x - c))

fits = []
gof = []
scalers = []
start_points = [
    (0.709754432349746, 0.910192467553578, 0.978691004156862),
    (0.515433736542118, 0.72193376784407, 0.955510096008974),
    (0.355142740203084, 0.49975884517448, 0.624609065987624),
    (0.45087821170349, 0.723648199208609, 0.253875154342024),
]

for i in range(num_neurons):
    x = xbins[i]
    y = ybins[i]
    # check variance to avoid degenerate fits
    if np.nanstd(x) < 1e-8 or np.nanstd(y) < 1e-8:
        fits.append((np.nan, np.nan, np.nan))
        gof.append({'rsquare': np.nan})
        scalers.append((0, 1))
        continue

    # z-score x for numeric stability; store mean/std to rescale during prediction
    xm = np.nanmean(x)
    xs = np.nanstd(x)
    x_scaled = (x - xm) / xs
    scalers.append((xm, xs))
    try:
        popt, pcov = curve_fit(sigmoid, x_scaled, y, p0=start_points[i], maxfev=20000)
        fits.append(popt)
        # compute R^2-like measure
        residuals = y - sigmoid(x_scaled, *popt)
        ss_res = np.nansum(residuals ** 2)
        ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        gof.append({'rsquare': r_squared})
    except Exception as e:
        fits.append((np.nan, np.nan, np.nan))
        gof.append({'rsquare': np.nan})

import pickle
with open('fits.pkl', 'wb') as f:
    pickle.dump({'fits': fits, 'gof': gof}, f)

# predictions on test segment 15001:20000 (python indices 15000:20000)
pred_test = np.zeros((num_neurons, 5000))
for i in range(num_neurons):
    conv_res = np.convolve(stimulus[15000:20000], h[i, :])
    conv_res = conv_res[:5000]
    a, b, c = fits[i]
    xm, xs = scalers[i] if i < len(scalers) else (0, 1)
    if np.isnan(a):
        pred_test[i, :] = conv_res
    else:
        # scale conv_res using fitted scaler
        conv_scaled = (conv_res - xm) / xs
        pred_test[i, :] = sigmoid(conv_scaled, a, b, c)

gt = np.zeros((num_neurons, 5000))
for i in range(num_neurons):
    gt[i, :] = 1000.0 * psth[i, 15000:20000]

"""
----------- Question 6: R^2 analysis and pruning ----------
"""
R_squared = np.zeros(num_neurons)
def safe_r_squared(y_true, y_pred):
    # handle NaNs
    if np.isnan(y_true).all() or np.isnan(y_pred).all():
        return np.nan
    # compute std ignoring NaNs
    std_true = np.nanstd(y_true)
    std_pred = np.nanstd(y_pred)
    if std_true == 0 or std_pred == 0:
        # no variance in one of the signals -> correlation undefined, return 0.0
        return 0.0
    # compute Pearson r safely
    try:
        r = np.corrcoef(np.nan_to_num(y_true), np.nan_to_num(y_pred))[0, 1]
        return r ** 2
    except Exception:
        return np.nan

for i in range(num_neurons):
    R_squared[i] = safe_r_squared(gt[i, :], pred_test[i, :])
    print(f'Neuron {i+1} R^2 = {R_squared[i]}')

# Iterative pruning for neurons 2 and 3 (indices 1 and 2)
def prune_h_and_track(h_row, fit_params, gt_row, stimulus_segment, max_iters=100):
    A = []
    B = []
    h_copy = h_row.copy()
    old_r_sq = np.nan
    # initial
    conv_res = np.convolve(stimulus_segment, h_copy)[:5000]
    a, b, c = fit_params
    if np.isnan(a):
        pred = conv_res
    else:
        pred = a / (1 + np.exp(-b * (conv_res - c)))
    old_r_sq = safe_r_squared(gt_row, pred)
    count = 0
    while count < max_iters:
        # find smallest non-zero abs coeff
        nonzero_idx = np.where(np.abs(h_copy) > 0)[0]
        if nonzero_idx.size == 0:
            break
        idx = nonzero_idx[np.argmin(np.abs(h_copy[nonzero_idx]))]
        h_copy[idx] = 0
        conv_res = np.convolve(stimulus_segment, h_copy)[:5000]
        if np.isnan(a):
            pred = conv_res
        else:
            pred = a / (1 + np.exp(-b * (conv_res - c)))
        new_r_sq = safe_r_squared(gt_row, pred)
        count += 1
        A.append(count)
        B.append(new_r_sq)
        # stop if improvement is small or negative
        if (old_r_sq - new_r_sq) >= 0.01:
            break
        old_r_sq = new_r_sq
    return np.array(A), np.array(B)

max_corr_coef_2 = np.nan
max_corr_coef_3 = np.nan
for neuron_idx in [1, 2]:
    A, B = prune_h_and_track(h[neuron_idx, :].copy(), fits[neuron_idx], gt[neuron_idx, :], stimulus[15000:20000])
    plt.figure()
    plt.scatter(A, B)
    plt.title(f'Plot of prediction performance with iterations for neuron {neuron_idx+1}')
    plt.show()
    if B.size:
        if neuron_idx == 1:
            max_corr_coef_2 = np.nanmax(B)
        elif neuron_idx == 2:
            max_corr_coef_3 = np.nanmax(B)

"""
----------- FFTs and final plots ----------
"""
plt.figure()
for i in range(num_neurons):
    plt.subplot(2, 2, i + 1)
    plt.stem(h[i, :])
    plt.title(f'Linear filter for Neuron {i+1}')
plt.show()

f_t = np.fft.fft(h, axis=1)
vect = np.arange(-50, 50)
plt.figure()
for i in range(num_neurons):
    plt.subplot(2, 2, i + 1)
    plt.plot(vect, np.abs(f_t[i, :len(vect)]))
    plt.title(f'FFT of filter for Neuron {i+1}')
plt.show()

print(f'maximum predicted performance for neuron2 is {max_corr_coef_2}')
print(f'maximum predicted performance for neuron3 is {max_corr_coef_3}')

"""
----------- Question 7: Mutual information (port of MATLAB functions) ----------
Note: this section may be computationally heavy; functions implemented below.
"""

def VPSDM(tli, tlj, q):
    nspi = len(tli)
    nspj = len(tlj)
    if q == 0:
        return abs(nspi - nspj)
    if q == np.inf:
        return nspi + nspj
    scr = np.zeros((nspi + 1, nspj + 1))
    scr[:, 0] = np.arange(0, nspi + 1)
    scr[0, :] = np.arange(0, nspj + 1)
    if nspi and nspj:
        for ii in range(1, nspi + 1):
            for jj in range(1, nspj + 1):
                scr[ii, jj] = min(scr[ii - 1, jj] + 1,
                                 scr[ii, jj - 1] + 1,
                                 scr[ii - 1, jj - 1] + q * abs(tli[ii - 1] - tlj[jj - 1]))
    return scr[nspi, nspj]

def meandist(i_idx, rep, spike_segments, q):
    mean_dist = np.zeros(8)
    for i1 in range(8):
        for rep1 in range(50):
            if (rep1 == rep and i1 == i_idx):
                continue
            mean_dist[i1] += VPSDM(spike_segments[i_idx][rep], spike_segments[i1][rep1], q)
    mean_dist = mean_dist / 50.0
    return mean_dist

def MutualInfo(confusion):
    MI = 0.0
    K = confusion.shape[0]
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            if confusion[i, j] != 0:
                MI += confusion[i, j] / K * np.log2(confusion[i, j] / confusion[:, j].sum())
    return MI

def ci90(ci, MI):
    MIbar = MI.mean(axis=1)
    MIstd = np.std(np.abs(MI), axis=1, ddof=0)
    alpha = 1 - ci
    from scipy.stats import t
    T_multiplier = t.ppf(1 - alpha / 2, 99)
    ci90 = T_multiplier * MIstd / np.sqrt(99)
    return ci90

print('Completed porting Questions 2-7 (approx).')
