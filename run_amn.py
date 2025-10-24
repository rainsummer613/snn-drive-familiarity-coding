import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import exp_name, params_fixed, combinations, jitter_window, n_neurons
from src.utils import build_input_one_to_one, generate_lgn_inputs, generate_lgn_inputs_sync
from src.network_abstract import build_connectivity
from src.model import NetworkLIF
from src.measure import measure_rsync

np.set_printoptions(suppress=True)
rs = 42

def fam_barplot(performances, plot_name):
    # Example performance scores for 7 models (e.g., accuracy, F1-score, etc.)
    models = ["I", "M", "A",
              "I", "M", "A",
              ""]
    performance = list(performances) + [0.33]

    # Define colors and transparency levels
    base_colors = ["teal", "teal", "teal", "orangered", "orangered", "orangered", "black"]
    alphas = [0.65, 0.35, 0.15, 0.65, 0.35, 0.15, 0.2]  # Adjust transparency

    # Convert base colors into RGBA format with different transparencies
    colors = [to_rgba(color, alpha) for color, alpha in zip(base_colors, alphas)]

    # Define group positions (grouping green, orange, and grey separately)
    group_positions = [0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0]  # Keeping groups together with spacing
    group_names = ["Spike count", "Rsync", "Baseline"]
    group_x_positions = [0.25, 1.25, 2.1]

    # Create bar plot with custom width and grouped bars
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_width = 0.2  # Adjust bar width

    bars = ax.bar(group_positions, performance, color=colors, edgecolor=base_colors,
                  linewidth=1.25, width=bar_width)

    ax.axhline(y=performance[-1], color="black", linestyle="dashed", linewidth=1.5, label="Model 7 Threshold")

    # Add group names below the x-ticks, centered under each group
    for pos, group_name in zip(group_x_positions, group_names):
        ax.text(pos, -0.175, group_name, ha="center", fontsize=20, fontweight="bold")

    # Set x-axis labels correctly under grouped bars
    ax.set_xticks(group_positions, models)  # rotation=40, ha="right")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    # Labels and title
    ax.set_ylabel("F1 Score", fontsize=24)
    ax.set_ylim(0, 1.02)

    fig.savefig(plot_name)

def process_ratio(queue,
                  model_class, model_params, n_neurons, w_ff,
                  length, input_rates, iterations, n_patterns, pattern_active, pattern_size, n_sources, plot_dir,
                  w_exc_p, w_som_p):
    """Worker function to process a single w_exc_intra_cluster_ratio."""
    results = {"f1_fam": {
                    "Ratio": pattern_active/pattern_size,
                    "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
                    "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],
                    },
              "fp_fn": {
                    "Ratio": pattern_active/pattern_size,
                    "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
                    "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],
                    },
              "f1_class": {
                    "Ratio": pattern_active/pattern_size,
                    "Max pattern Rsync": [], "Max pattern SC": [],
                    "Overlap Familiar": [],
                    "Overlap Novel": []
                    },
              "completion": {
                    "Ratio": pattern_active/pattern_size,
                    "Above 50%": [],
                    "Above 95%": []
                    },
              "stats": {
                    "Ratio": pattern_active/pattern_size,
                    "Rsync Fam": [],
                    "Rsync New": [],
                    "SC Fam": [],
                    "SC New": []
              }
                }

    # build connectivity
    for iter in range(10):
        patterns, patterns_new, w_exc, w_som, w_pv = build_connectivity(n_neurons=n_neurons,
                                                          n_patterns=n_patterns,
                                                          pattern_size=pattern_size,
                                                          w_exc_p=w_exc_p, w_som_p=w_som_p)
        w_inh = (w_som + w_pv * 0)
        patterns_all = patterns + patterns_new
        pattern_fam_ids = range(0, len(patterns))
        pattern_new_ids = range(len(patterns), len(patterns) + len(patterns_new))

        print(f"--- {pattern_active/100} :: {iter}  ---")
        print(f"EXPERIMENT PARAMETERS :: n {n_patterns}, exc {w_exc_p}, inh {w_som_p}")
        print("---")
        print(f"EXC :: sum {round(w_exc.sum(1).mean(), 3)}, mean {round(w_exc[w_exc > 0].mean(), 3)}, max {round(w_exc.max(), 3)}, num {round(np.count_nonzero(w_exc, axis=1).mean(), 3)}")
        print(f"INH :: sum {round(w_inh.sum(1).mean(), 3)}, mean {round(w_inh[w_inh < 0].mean(), 3)}, min {round(w_inh.min(), 3)}, num {round(np.count_nonzero(w_inh, axis=1).mean(), 3)}")
        w_all = w_exc + w_inh
        print(f"ALL :: sum {round(w_all.sum(1).mean(), 3)}, mean {round(w_all.mean(), 3)}, min {round(w_all.min(), 3)}, max {round(w_all.max(), 3)}")
        print("---")
        y_class_sc, y_class_rsync, y_class, y_fam = [], [], [], []
        x_rsync_input, x_rsync_max_sc, x_rsync_max_rsync, x_rsync_all = [], [], [], []
        x_sc_input, x_sc_max_sc, x_sc_max_rsync, x_sc_all = [], [], [], []
        sc_overlap_new, sc_overlap_fam = [], []

        completed_above_50, completed_above_95 = [], []

        model = model_class(n_neurons=n_neurons, n_inputs=n_neurons,
                            w_exc=w_exc, w_inh=w_inh, w_ff=w_ff,
                            **model_params)

        for input_rate in input_rates:
            #print(input_rate)
            for iteration in range(iterations):
                pattern_fam_idx = np.random.choice(pattern_fam_ids)
                pattern_new_idx = np.random.choice(pattern_new_ids)

                pattern_fam = patterns_all[pattern_fam_idx]
                pattern_new = patterns_all[pattern_new_idx]

                pattern_fam_cur = sorted(np.random.choice(pattern_fam, pattern_active, replace=False))
                to_complete = list(set(pattern_fam) - set(pattern_fam_cur))

                #patterns_all = patterns + [pattern_new]

                for sample_fam, (sample_class, sample_input) in {0: (pattern_new_idx, pattern_new), 1: (pattern_fam_idx, pattern_fam_cur)}.items():
                    if jitter_window is None:
                        ext_input = generate_lgn_inputs(n_neurons=n_neurons, n_sources=n_sources, pattern=sample_input, input_rate=input_rate, length=length + 1)
                    else:
                        ext_input = generate_lgn_inputs_sync(n_neurons=n_neurons, n_sources=n_sources, pattern=sample_input, input_rate=input_rate, length=length + 1, jitter_window=jitter_window)

                    voltage, spikes = model.simulate(length=length, external_input=ext_input)

                    top_100_ids = np.argsort(spikes.sum(1))[-100:][::-1]
                    overlap = len(np.intersect1d(top_100_ids, sample_input)) / 100
                    if sample_fam == 0:
                        sc_overlap_new.append(overlap)
                    else:
                        sc_overlap_fam.append(overlap)

                    all_sc = spikes.sum(axis=1)

                    sample_sc = [np.mean(all_sc[p]) for p in patterns_all]  # [all_sc[p].mean() for p in patterns_all]
                    sample_sc = [0.0 if np.isnan(val) else val for val in sample_sc]
                    max_sc_id = np.argmax(sample_sc)

                    sample_rsync = [measure_rsync(spikes[p]) for p in patterns_all]
                    sample_rsync = [0.0 if np.isnan(val) else val for val in sample_rsync]
                    max_rsync_id = np.argmax(sample_rsync)

                    x_sc_input.append(all_sc[sample_input].mean())
                    x_sc_all.append(all_sc[spikes.sum(1) > 0].mean())
                    x_sc_max_rsync.append(sample_sc[max_rsync_id])
                    x_sc_max_sc.append(sample_sc[max_sc_id])

                    if sample_fam == 1:
                        if len(to_complete) == 0:
                            completed_above_50.append(1.0)
                            completed_above_95.append(1.0)
                        else:
                            non_pattern = np.delete(all_sc, pattern_fam)
                            non_pattern = non_pattern[non_pattern > 0]
                            if len(non_pattern) == 0:
                                completed_above_50.append(0.0)
                                completed_above_95.append(0.0)
                            else:
                                completed_above_50.append(np.sum(all_sc[to_complete] > np.percentile(non_pattern, 50)) / len(to_complete))
                                completed_above_95.append(np.sum(all_sc[to_complete] > np.percentile(non_pattern, 95)) / len(to_complete))

                    x_rsync_input.append(measure_rsync(spikes[sample_input]))
                    x_rsync_all.append(measure_rsync(spikes[spikes.sum(1) > 0]))
                    x_rsync_max_rsync.append(sample_rsync[max_rsync_id])
                    x_rsync_max_sc.append(sample_rsync[max_sc_id])

                    y_class_sc.append(max_sc_id)
                    y_class_rsync.append(max_rsync_id)
                    y_class.append(sample_class)
                    y_fam.append(sample_fam)

        baseline_class = np.random.choice(np.arange(1, n_patterns + 2), size=len(y_class))
        baseline_fam = np.random.choice([0, 1], size=len(y_fam))

        x_sc_input = np.array(x_sc_input)
        x_rsync_input = np.array(x_rsync_input)
        y_fam = np.array(y_fam)
        results["stats"]["SC Fam"].append(round(x_sc_input[y_fam == 1].mean(), 3))
        results["stats"]["SC New"].append(round(x_sc_input[y_fam == 0].mean(), 3))
        results["stats"]["Rsync Fam"].append(round(x_rsync_input[y_fam == 1].mean(), 3))
        results["stats"]["Rsync New"].append(round(x_rsync_input[y_fam == 0].mean(), 3))

        results["f1_class"]["Max pattern Rsync"].append(round(f1_score(y_class, y_class_rsync, average="macro"), 3))
        results["f1_class"]["Max pattern SC"].append(round(f1_score(y_class, y_class_sc, average="macro"), 3))

        results["f1_class"]["Overlap Familiar"].append(round(np.mean(sc_overlap_fam), 3))
        results["f1_class"]["Overlap Novel"].append(round(np.mean(sc_overlap_new), 3))

        #print(f"{pattern_active/100} PATTERN COMPLETION: {np.mean(completed_above_50)}")
        results["completion"]["Above 50%"].append(round(np.mean(completed_above_50), 3))
        results["completion"]["Above 95%"].append(round(np.mean(completed_above_95), 3))

        #print(f"{pattern_active/100} FAMILIARITY")
        performances = []
        for x_key, x_val in {"SC Input": x_sc_input, "SC Max SC": x_sc_max_sc, "SC Max Rsync": x_sc_max_rsync, "SC Total": x_sc_all,
                             "Rsync Input": x_rsync_input, "Rsync Max SC": x_rsync_max_sc, "Rsync Max Rsync": x_rsync_max_rsync, "Rsync Total": x_rsync_all,
                             }.items():

            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=rs)
            f1_scores = []
            fp_fn_ratios = []

            X = np.array(x_val).reshape(-1, 1)
            y = np.array(y_fam)

            for train_index, val_index in skf.split(X, y):  # Stratified splits
                # Split into training and validation sets
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Train model using F1-score
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_val_pred)
                f1_scores.append(f1)

                # Evaluate on validation set using FPR - FNR
                fp = np.sum((y_val_pred == 1) & (y_val == 0))
                fn = np.sum((y_val_pred == 0) & (y_val == 1))
                tp = np.sum((y_val_pred == 1) & (y_val == 1))
                tn = np.sum((y_val_pred == 0) & (y_val == 0))
                fp_fn_ratio = fp / (fp + tn + 0.0001) - fn / (fn + tp + 0.0001)
                fp_fn_ratios.append(fp_fn_ratio)

            results["f1_fam"][x_key].append(round(np.mean(f1_scores), 3))
            results["fp_fn"][x_key].append(round(np.mean(fp_fn_ratios), 3))

            if x_key not in ("SC Max Rsync", "Rsync Max Rsync"):
                performances.append(np.mean(f1_scores))

        plot_path = f"{plot_dir}/{round(pattern_active/100, 1)}.png"
        fam_barplot(performances, plot_path)

        print(f"RES :: {results}")
        print()
        print()

    filtered = {k: v for k, v in results.items() if k != 'fp_fn'}
    combined = {}
    for subdict in filtered.values():
        for key, val in subdict.items():
            if key != 'Ratio':
                combined[key] = val
    df = pd.DataFrame(combined)
    df.to_csv(f"{plot_dir}/{pattern_active / pattern_size}.csv", index=False)
    queue.put(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--combination', type=int, default=0)
    args = parser.parse_args()

    n_patterns, w_exc_p, w_som_p = combinations[args.combination]
    w_pv_p = 0.0

    length = 500
    model_class = NetworkLIF
    n_sources = 1
    rate_mean = 50

    # parameters: CONNECTIVITY
    w_ff_val = 3
    pattern_size = 100
    w_exc_intra_cluster_ratio = 1.0

    # parameters: EXPERIMENT
    rate_var = 40
    model_params = params_fixed.copy()

    if 1 <= n_patterns <= 500:
        x_sc = []
        x_rsync = []
        y = []

        ks_stat_dict = {"sc": [], "rsync": []}
        ks_p_dict = {"sc": [], "rsync": []}
        kl_dict = {"sc": [], "rsync": []}

        fig, ax  = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)
        plot_dir = f"data/{exp_name}/{w_exc_p}_{w_som_p}_{w_pv_p}/size_100/n_{n_patterns}"
        if jitter_window:
            plot_dir = f"data/{exp_name}/{jitter_window}/{w_exc_p}_{w_som_p}_{w_pv_p}/size_100/n_{n_patterns}"
        os.makedirs(plot_dir, exist_ok=True)

        min_rate, max_rate = rate_mean - rate_var, rate_mean + rate_var
        input_rates = np.arange(int(min_rate), int(max_rate), 1)
        n_samples = 300
        iterations = max(1, n_samples // len(input_rates))

        manager = mp.Manager()
        queue = manager.Queue()
        processes = []
        f1_fam = {
            "Ratio": [],
            "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
            "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],

            "Rsync Input SD": [], "Rsync Max Rsync SD": [], "Rsync Max SC SD": [], "Rsync Total SD": [],
            "SC Input SD": [], "SC Max Rsync SD": [], "SC Max SC SD": [], "SC Total SD": [],
        }
        fp_fn_fam = {
            "Ratio": [],
            "Rsync Input": [], "Rsync Max Rsync": [], "Rsync Max SC": [], "Rsync Total": [],
            "SC Input": [], "SC Max Rsync": [], "SC Max SC": [], "SC Total": [],

            "Rsync Input SD": [], "Rsync Max Rsync SD": [], "Rsync Max SC SD": [], "Rsync Total SD": [],
            "SC Input SD": [], "SC Max Rsync SD": [], "SC Max SC SD": [], "SC Total SD": [],
        }
        f1_class = {
            "Ratio": [],
            "Max pattern Rsync": [], "Max pattern SC": [],
            "Overlap Familiar": [], "Overlap Novel": [],

            "Max pattern Rsync SD": [], "Max pattern SC SD": [],
            "Overlap Familiar SD": [], "Overlap Novel SD": [],
        }
        completion = {
            "Ratio": [],
            "Above 50%": [], "Above 95%": [],
            "Above 50% SD": [], "Above 95% SD": [],
        }
        stats = {
            "Ratio": [],
            "Rsync Fam": [], "Rsync New": [], "SC Fam": [], "SC New": [],
            "Rsync Fam SD": [], "Rsync New SD": [], "SC Fam SD": [], "SC New SD": [],
        }

        # Feedforward input connections
        w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val

        #for w_exc_intra_cluster_ratio in np.arange(0.1, 1.1, 0.1):
        for pattern_active in (100, 90, 80, 70, 60):
            p = mp.Process(target=process_ratio, args=(queue, model_class, model_params,
                                   n_neurons, w_ff, length, input_rates, iterations,
                                   n_patterns, pattern_active, pattern_size, n_sources, plot_dir,
                                   w_exc_p, w_som_p))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        while not queue.empty():
            result = queue.get()
            for key in f1_fam:
                if "SD" not in key and key != "Ratio":
                    res = result["f1_fam"][key]
                    if isinstance(res, list):
                        f1_fam[key].append(np.round(np.mean(res), 3))
                        f1_fam[f"{key} SD"].append(np.round(np.std(res), 3))
            f1_fam["Ratio"].append(result["f1_fam"]["Ratio"])

            for key in f1_class:
                if "SD" not in key and key != "Ratio":
                    res = result["f1_class"][key]
                    if isinstance(res, list):
                        f1_class[key].append(np.round(np.mean(res), 3))
                        f1_class[f"{key} SD"].append(np.round(np.std(res), 3))
            f1_class["Ratio"].append(result["f1_class"]["Ratio"])

            for key in completion:
                if "SD" not in key and key != "Ratio":
                    res = result["completion"][key]
                    if isinstance(res, list):
                        completion[key].append(np.round(np.mean(res), 3))
                        completion[f"{key} SD"].append(np.round(np.std(res), 3))
            completion["Ratio"].append(result["completion"]["Ratio"])

            for key in stats:
                if "SD" not in key and key != "Ratio":
                    res = result["stats"][key]
                    if isinstance(res, list):
                        stats[key].append(np.round(np.mean(res), 3))
                        stats[f"{key} SD"].append(np.round(np.std(res), 3))
            stats["Ratio"].append(result["stats"]["Ratio"])

        # Save raw stats
        df = pd.DataFrame(stats)
        df = df.sort_values(by=["Ratio"], ascending=True)
        df.to_csv(f"{plot_dir}/stats.csv", index=False)

        # Plot completion results
        df = pd.DataFrame(completion)
        df = df.sort_values(by=["Ratio"], ascending=True)
        df.to_csv(f"{plot_dir}/completion.csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 5), nrows=1, ncols=1)
        ax.plot(df["Ratio"], df["Above 95%"], color="darkslategrey", label="Above 95%")
        ax.fill_between(
            np.arange(len(df["Above 95%"])),
            df["Above 95%"] - df["Above 95% SD"],
            df["Above 95%"] + df["Above 95% SD"],
            alpha=0.3,
        )
        ax.plot(df["Ratio"], df["Above 50%"], color="cadetblue", label="Above 50%")
        ax.fill_between(
            df["Ratio"],
            df["Above 50%"] - df["Above 50% SD"],
            df["Above 50%"] + df["Above 50% SD"],
            alpha=0.3,
        )
        ax.set_title("Pattern completion", fontsize=20)
        ax.set_xlabel("Input proportion", fontsize=16)
        ax.set_ylabel("Completed proportion", fontsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=16)
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/completion.png")
        fig.savefig(f"{plot_dir}/completion.svg")

        # Plot Classification results: F1 and max=pattern proportion
        df = pd.DataFrame(f1_class)
        df = df.sort_values(by=["Ratio"], ascending=True)
        df.to_csv(f"{plot_dir}/class.csv", index=False)

        # Classification performance
        #baseline_class = np.random.choice(np.arange(1, n_patterns + 2), size=1000)

        fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
        ax[0].plot(df["Ratio"], df["Max pattern Rsync"], color="orangered", label="Rsync")
        ax[0].fill_between(
            df["Ratio"],
            df["Max pattern Rsync"] - df["Max pattern Rsync SD"],
            df["Max pattern Rsync"] + df["Max pattern Rsync SD"],
            alpha=0.3,
        )
        ax[0].plot(df["Ratio"], df["Max pattern SC"], color="teal", label="Spike count")
        ax[0].fill_between(
            df["Ratio"],
            df["Max pattern SC"] - df["Max pattern SC SD"],
            df["Max pattern SC"] + df["Max pattern SC SD"],
            alpha=0.3,
        )
        ax[0].set_title("Pattern classification: Top SC", fontsize=20)
        # ax[0].set_xlabel("Intra-pattern connections", fontsize=16)
        ax[0].set_ylabel("F1 score", fontsize=18)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].set_ylim(0.0, 1.05)
        #ax[0].axhline(y=0.012, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[0].legend(fontsize=16)

        ax[1].plot(df["Ratio"], df["Overlap Familiar"], color="cadetblue", label="Familiar")
        ax[1].fill_between(
            df["Ratio"],
            df["Overlap Familiar"] - df["Overlap Familiar SD"],
            df["Overlap Familiar"] + df["Overlap Familiar SD"],
            alpha=0.3,
        )
        ax[1].plot(df["Ratio"], df["Overlap Novel"], color="cadetblue", linestyle="dotted", linewidth=1.75, label="Novel")
        ax[1].fill_between(
            df["Ratio"],
            df["Overlap Novel"] - df["Overlap Novel SD"],
            df["Overlap Novel"] + df["Overlap Novel SD"],
            alpha=0.3,
        )
        ax[1].set_title("Input & correct overlap", fontsize=20)
        ax[1].set_xlabel("Input proportion", fontsize=18)
        ax[1].set_ylabel("Overlap", fontsize=18)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylim(0.0, 1.05)
        ax[1].axhline(y=0.125, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[1].legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary class.png")
        fig.savefig(f"{plot_dir}/summary class.svg")

        # Summary
        fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
        ax[0].plot(df["Ratio"], df["Max pattern Rsync"], color="orangered", label="Rsync")
        ax[0].fill_between(
            df["Ratio"],
            df["Max pattern Rsync"] - df["Max pattern Rsync SD"],
            df["Max pattern Rsync"] + df["Max pattern Rsync SD"],
            alpha=0.3,
        )
        ax[0].plot(df["Ratio"], df["Max pattern SC"], color="teal", label="Spike count")
        ax[0].fill_between(
            df["Ratio"],
            df["Max pattern SC"] - df["Max pattern SC SD"],
            df["Max pattern SC"] + df["Max pattern SC SD"],
            alpha=0.3,
        )
        ax[0].set_title(f"Pattern classification", fontsize=20)
        #ax[0].set_xlabel("Intra-pattern connections", fontsize=18)
        ax[0].set_ylabel("Weighted F1 score", fontsize=18)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].set_ylim(0.0, 1.05)
        #ax[0].legend(fontsize=16)

        df = pd.DataFrame(f1_fam)
        df = df.sort_values(by=["Ratio"], ascending=True)
        ax[1].plot(df["Ratio"], df["Rsync Max SC"], color="orangered", label="Rsync")
        ax[1].fill_between(
            df["Ratio"],
            df["Rsync Max SC"] - df["Rsync Max SC SD"],
            df["Rsync Max SC"] + df["Rsync Max SC SD"],
            alpha=0.3,
        )
        ax[1].plot(df["Ratio"], df["SC Max SC"], color="teal", label="Spike count")
        ax[1].fill_between(
            df["Ratio"],
            df["SC Max SC"] - df["SC Max SC SD"],
            df["SC Max SC"] + df["SC Max SC SD"],
            alpha=0.3,
        )
        ax[1].set_title("Familiarity detection", fontsize=20)
        ax[1].set_xlabel("Input proportion", fontsize=18)
        ax[1].set_ylabel("F1 score", fontsize=18)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylim(0, 1.05)
        ax[1].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[1].legend(fontsize=16)

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary.png")
        fig.savefig(f"{plot_dir}/summary.svg")

        # Plot Familiarity results: F1 and error analysis
        label = "F1 score"
        values, lims = f1_fam, (0.0, 1.05)

        df = pd.DataFrame(values)
        df = df.sort_values(by=["Ratio"], ascending=True)
        df.to_csv(f"{plot_dir}/fam {label}.csv", index=False)

        fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)
        ax[0].plot(df["Ratio"], df["Rsync Input"], color="orangered", label="Rsync")
        ax[0].fill_between(
            df["Ratio"],
            df["Rsync Input"] - df["Rsync Input SD"],
            df["Rsync Input"] + df["Rsync Input SD"],
            alpha=0.3,
        )
        ax[0].plot(df["Ratio"], df["SC Input"], color="teal", label="Spike count")
        ax[0].fill_between(
            df["Ratio"],
            df["SC Input"] - df["SC Input SD"],
            df["SC Input"] + df["SC Input SD"],
            alpha=0.3,
        )
        ax[0].set_title("Stimulated neurons", fontsize=20)
        #ax[0].set_xlabel("Intra-pattern connections", fontsize=16)
        ax[0].set_ylabel(label, fontsize=18)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].tick_params(axis='x', labelsize=16)
        ax[0].tick_params(axis='y', labelsize=16)
        ax[0].set_ylim(lims[0], lims[1])
        if label == "F1 score":
            ax[0].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
        ax[0].legend(fontsize=16)


        ax[1].plot(df["Ratio"], df["Rsync Max SC"], color="orangered", label="Rsync")
        ax[1].fill_between(
            df["Ratio"],
            df["Rsync Max SC"] - df["Rsync Max SC SD"],
            df["Rsync Max SC"] + df["Rsync Max SC SD"],
            alpha=0.3,
        )
        ax[1].plot(df["Ratio"], df["SC Max SC"], color="teal", label="Spike count")
        ax[1].fill_between(
            df["Ratio"],
            df["SC Max SC"] - df["SC Max SC SD"],
            df["SC Max SC"] + df["SC Max SC SD"],
            alpha=0.3,
        )
        ax[1].set_title("Top SC pattern", fontsize=20)
        ax[1].set_xlabel("Input proportion", fontsize=18)
        ax[1].set_ylabel(label, fontsize=18)
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].tick_params(axis='x', labelsize=16)
        ax[1].tick_params(axis='y', labelsize=16)
        ax[1].set_ylim(lims[0], lims[1])
        ax[1].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")

        fig.tight_layout()
        fig.savefig(f"{plot_dir}/summary {label}.png")
        fig.savefig(f"{plot_dir}/summary {label}.svg")
        