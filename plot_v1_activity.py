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
from src.network_v1 import build_connectivity as build_connectivity_v1
from src.config import params_fixed
from src.measure import measure_rsync

def process_comb(queue, length, n_neurons, n_sources, model, input_rate, stimulus, i):
    poisson_input = generate_lgn_inputs(n_neurons, n_sources, stimulus, input_rate, length + 1)
    voltage, spikes = model.simulate(length=length, external_input=poisson_input)
    print("DONE", input_rate, spikes.sum(1).mean())
    queue.put({"idx": i, "spikes": spikes})

if __name__ == "__main__":
    plot_dir = "data/v1"
    length = 500
    n_sources = 1

    grid_size = 10
    n_neurons = grid_size * grid_size * 8
    pattern_size = 40
    n_groups = int(pattern_size / 4)
    n_max = 0

    w_ff_val = 4

    w_exc, w_som, w_pv = build_connectivity_v1(grid_size)
    w_inh = w_pv + w_som * 0
    w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val

    print("w_exc", w_exc.sum(1).mean(), w_exc[w_exc > 0].mean(), w_exc.max())
    print("w_inh", w_inh.sum(1).mean(), w_inh[w_inh < 0].mean(), w_inh.min())
    w_all = w_exc + w_inh
    print("w_all", w_all.sum(1).mean(), w_all.mean(), w_all.min(), w_all.max())

    rates = (70, 30)  # saliency
    pattern_new = sorted(np.random.choice(np.arange(n_neurons), pattern_size, replace=False))

    pattern_orig = list(np.arange(200, 200+pattern_size))
    stimuli = (pattern_orig, pattern_new)  # familiarity
    #print("FAMILIAR", w_exc[pattern_orig, :][:, pattern_orig].sum())
    #print("NEW", w_exc[pattern_new, :][:, pattern_new].sum())

    combs = ((rates[0], stimuli[0]), (rates[0], stimuli[1]), (rates[1], stimuli[0]), (rates[1], stimuli[1]))

    model = NetworkLIF(n_neurons=n_neurons, n_inputs=n_neurons,
                            w_exc=w_exc, w_inh=w_inh, w_ff=w_ff,
                            **params_fixed)

    manager = mp.Manager()
    queue = manager.Queue()
    processes = []

    for i, c in enumerate(combs):
        input_rate, stimulus = c
        p = mp.Process(target=process_comb, args=(queue, length, n_neurons, n_sources, model, input_rate, stimulus, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    results = [0, 0, 0, 0]
    while not queue.empty():
        result = queue.get()
        results[result["idx"]] = result["spikes"]

    labels_y = ["Salient input", "Weak input"]
    labels_x = ["Familiar input", "New input"]

    fig, ax = plt.subplots(figsize=(5, 5), nrows=2, ncols=2)

    neurons_print = 40
    counter = 0
    for row_i in range(2):  # saliency
        fig.text(0.04, 0.73 - row_i * 0.43, labels_y[row_i], va='center', ha='right', fontsize=14, rotation=90)

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
                0.96, 0.04,
                text,
                transform=ax[row_i, col_i].transAxes,
                fontsize=13,
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
            ax[row_i, col_i].scatter(spike_x, spike_y, color=spike_colors, s=3)
            ax[row_i, col_i].set_xlim(0, length)
            ax[row_i, col_i].set_ylim(0.5, neurons_print + 0.5)

            if col_i == 0:
                ax[row_i, col_i].set_ylabel("Input neurons", fontsize=15, labelpad=6)
            else:
                ax[row_i, col_i].set_yticks([])

            if row_i == 1:
                ax[row_i, col_i].set_xlabel("Time (ms)", fontsize=15, labelpad=10)
            else:
                ax[row_i, col_i].set_xticks([])

            if row_i == 0:
                ax[row_i, col_i].set_title(labels_x[col_i], fontsize=17, pad=15)

            ax[row_i, col_i].tick_params(axis='both', labelsize=13)

            counter += 1

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    #fig.savefig(f"{plot_dir}/scatter_v1.png", dpi=500)
    #fig.savefig(f"{plot_dir}/scatter_v1.svg")
    plt.show()