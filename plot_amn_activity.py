import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.utils import build_input_one_to_one, generate_lgn_inputs
from src.model import NetworkLIF
from src.network_abstract import build_connectivity
from src.config import params_fixed
from src.measure import measure_rsync

def process_comb(queue, length, n_neurons, n_sources, model, input_rate, stimulus):
    poisson_input = generate_lgn_inputs(n_neurons, n_sources, stimulus, input_rate, length + 1)
    voltage, spikes = model.simulate(length=length, external_input=poisson_input)
    print("DONE", input_rate, spikes.sum(1).mean())
    queue.put(spikes)

if __name__ == "__main__":
    length = 500
    n_sources = 1

    n_neurons = 5000
    n_patterns = 300
    pattern_size = 100
    w_exc_p = 0.4
    w_som_p = 0.04

    # non-exp input: 5000, 0.3, 0.007
    # exp input:     1000, 0.5, 0.04; 5000, 0.5, 0.045; 5000, 0.4, 0.028 (0.6, 0.4); 0.4, 0.03 (0.6, 0.6)

    #grid_size = 10
    #n_neurons = grid_size * grid_size * 8
    #pattern_size = 24
    #n_groups = int(pattern_size / 4)
    #n_max = 0

    w_ff_val = 4

    patterns, _, w_exc, w_som, w_pv = build_connectivity(n_neurons=n_neurons, n_patterns=n_patterns,
                                                         pattern_size=pattern_size, w_exc_p=w_exc_p, w_som_p=w_som_p)
    #w_exc, w_som, w_pv = build_connectivity_v1(grid_size)
    w_inh = w_som + w_pv * 0
    w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val

    print("w_exc", w_exc.sum(1).mean(), w_exc[w_exc > 0].mean(), w_exc.max(), "num", np.count_nonzero(w_exc, axis=1).mean())
    print("w_inh", w_inh.sum(1).mean(), w_inh[w_inh < 0].mean(), w_inh.min(), "num", np.count_nonzero(w_inh, axis=1).mean())
    w_all = w_exc + w_inh
    print("w_all", w_all.sum(1).mean(), w_all.mean(), w_all.min(), w_all.max())

    rates = (80, 15)  # saliency
    pattern_new = sorted(np.random.choice(np.arange(n_neurons), pattern_size, replace=False))
    stimuli = (patterns[0], pattern_new)  # familiarity

    #print("FAMILIAR", w_exc[patterns[0], :][:, patterns[0]])
    #print("NEW", w_exc[pattern_new, :][:, pattern_new])

    #pattern_orig = sum(
    #    [list(np.arange(i, i + 4)) for i in np.arange(n_max, int(n_max + 10 * n_groups), 10)], [])
    #stimuli = (pattern_orig, pattern_new)  # familiarity

    combs = ((rates[0], stimuli[0]), (rates[0], stimuli[1]), (rates[1], stimuli[0]), (rates[1], stimuli[1]))
    model = NetworkLIF(n_neurons=n_neurons, n_inputs=n_neurons,
                            w_exc=w_exc, w_inh=w_inh, w_ff=w_ff,
                            **params_fixed)

    manager = mp.Manager()
    queue = manager.Queue()
    processes = []

    for c in combs:
        input_rate, stimulus = c
        p = mp.Process(target=process_comb, args=(queue, length, n_neurons, n_sources, model, input_rate, stimulus))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = []
    while not queue.empty():
        result = queue.get()
        results.append(result)

    labels_y = ["Salient input", "Weak input"]
    labels_x = ["Familiar input", "New input"]

    fig, ax = plt.subplots(figsize=(6, 6), nrows=2, ncols=2)

    neurons_print = 100
    counter = 0
    for row_i in range(2):  # saliency
        fig.text(0.04, 0.73 - row_i * 0.43, labels_y[row_i], va='center', ha='right', fontsize=20, rotation=90)

        for col_i in range(2):  # familiarity
            spike_trains = results[counter]
            input_indices = stimuli[col_i]

            spike_counts = spike_trains.sum(axis=1)

            top_indices = np.sort(np.argpartition(spike_counts, -neurons_print)[-neurons_print:])
            top_trains = spike_trains[top_indices]
            top_trains = top_trains[top_trains.sum(1) > 0]

            input_trains = spike_trains[input_indices]
            input_trains = input_trains[input_trains.sum(1) > 0]

            if len(top_trains) < 1:
                rsync = 0.0
                sc = 0
            else:
                rsync = round(measure_rsync(input_trains), 2)
                sc = int(round(input_trains.sum(1).mean()))

            text = f"Synchrony {rsync}\nSpike count {sc}"
            ax[row_i, col_i].text(
                0.97, 0.04,
                text,
                transform=ax[row_i, col_i].transAxes,
                fontsize=18,
                linespacing=1.5,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=1.0, edgecolor="white",
                          #boxstyle="round",
                          )
            )

            spike_x = []
            spike_y = []
            spike_colors = []

            for i, idx in enumerate(input_indices):
                spike_times = np.where(spike_trains[idx] == 1)[0]
                spike_x.extend(spike_times)
                spike_y.extend([i + 1] * len(spike_times))
                color = '#ce491e' if idx in input_indices else '#4a4848'
                spike_colors.extend([color] * len(spike_times))

            ax[row_i, col_i].set_facecolor('white')
            ax[row_i, col_i].scatter(spike_x, spike_y, color=spike_colors, s=1)
            ax[row_i, col_i].set_xlim(0, length)
            ax[row_i, col_i].set_ylim(0.5, neurons_print + 0.5)

            if col_i == 0:
                ax[row_i, col_i].set_ylabel("Input neurons", fontsize=18, labelpad=6)
            else:
                ax[row_i, col_i].set_yticks([])

            if row_i == 1:
                ax[row_i, col_i].set_xlabel("Time (ms)", fontsize=18, labelpad=10)
            else:
                ax[row_i, col_i].set_xticks([])

            if row_i == 0:
                ax[row_i, col_i].set_title(labels_x[col_i], fontsize=20, pad=15)

            ax[row_i, col_i].tick_params(axis='both', labelsize=16)

            counter += 1

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    #fig.savefig("scatter.png", dpi=500)
    #fig.savefig("scatter.svg")
    plt.show()