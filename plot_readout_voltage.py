import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.measure import measure_rsync

# -----------------------------
# Simulation parameters
# -----------------------------
dt = 1.0          # ms
T = 250.0         # ms
time = np.arange(0, T, dt)

rate = 20        # Hz
n_inputs = 100

# LIF parameters
tau_m = 20.0      # ms
v_rest = 0.0
v_reset = 0.0
v_th = 1e9        # no spiking/reset; we just measure voltage peak
w = 0.02          # input weight
tau_syn = 5.0     # ms

# Base spike train: identical source for all inputs before jitter
isi = 1000 / rate
base_spike_times = np.arange(isi, T, isi)


def make_jittered_spikes(jitter_ms):
    """
    Create n_inputs spike trains by jittering the same base spike times.
    jitter_ms = 0 gives perfectly synchronous input.
    """
    spikes = np.zeros((n_inputs, len(time)))

    for i in range(n_inputs):
        jittered_times = base_spike_times + np.random.uniform(-jitter_ms, jitter_ms, size=len(base_spike_times))
        jittered_times = jittered_times[(jittered_times >= 0) & (jittered_times < T)]

        idx = (jittered_times / dt).astype(int)
        spikes[i, idx] = 1

    return spikes

def run_lif(spikes):
    """
    One LIF neuron with exponential synaptic current.
    """
    v = np.zeros(len(time))
    syn = np.zeros(len(time))

    population_input = spikes.sum(axis=0)

    for t in range(1, len(time)):
        # synaptic current
        syn[t] = syn[t-1] + dt * (-syn[t-1] / tau_syn) + w * population_input[t]

        # membrane voltage
        dv = dt * (-(v[t-1] - v_rest) + syn[t]) / tau_m
        v[t] = v[t-1] + dv

        if v[t] >= v_th:
            v[t] = v_reset
    return v

if __name__ == "__main__":
    
    n_runs = 50
    jitter_values = np.linspace(0, 25, 25)
    
    all_rsync = np.zeros((n_runs, len(jitter_values)))
    all_peak = np.zeros((n_runs, len(jitter_values)))
    
    for run in range(n_runs):
        for i, jitter in enumerate(jitter_values):
            spikes = make_jittered_spikes(jitter)
            rsync = compute_rsync(spikes)
            v = run_lif(spikes)
    
            all_rsync[run, i] = rsync
            all_peak[run, i] = v.max()
    
    # Mean and SD
    rsync_mean = all_rsync.mean(axis=0)
    rsync_sd = all_rsync.std(axis=0)
    
    peak_mean = all_peak.mean(axis=0)
    peak_sd = all_peak.std(axis=0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(rsync_mean, peak_mean, color="orangered", lw=2)
    
    # SD band
    ax.fill_between(
        rsync_mean,
        peak_mean - peak_sd,
        peak_mean + peak_sd,
        color="orangered",
        alpha=0.25
    )
    
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Input Rsync", fontsize=22)
    ax.set_ylabel("Readout peak voltage", fontsize=22)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.2, 0.4)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.tight_layout()
    fig.savefig("Fig 8.svg"))
    plt.show()

    df = pd.DataFrame({
        "rsync_mean": rsync_mean,
        "rsync_sd": rsync_sd,
        "peak_mean": peak_mean,
        "peak_sd": peak_sd,
    })
    
    df.to_csv("results.csv", index=False)