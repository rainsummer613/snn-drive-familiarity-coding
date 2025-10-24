import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import seaborn as sns
import multiprocessing as mp

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import params_fixed
from src.utils import build_input_one_to_one, generate_lgn_inputs
from src.network_v1 import build_connectivity
from src.model import NetworkLIF
from src.measure import measure_rsync

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

np.set_printoptions(suppress=True)

from scipy.stats import ks_2samp, entropy
from scipy.stats import gaussian_kde

def bc_coefficient(sample1, sample2):
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    sample1 /= np.sum(sample1)
    sample2 /= np.sum(sample2)

    return np.sum(np.sqrt(sample1 * sample2))

def kolmogorov_smirnov_test(sample1, sample2):
    """
    Perform KS test to check if two samples come from the same distribution.
    Returns the KS statistic and p-value.
    """
    ks_stat, p_value = ks_2samp(sample1, sample2)
    return ks_stat, p_value

def symmetric_kl_divergence(sample1, sample2, bandwidth=0.1, epsilon=1e-10):
    """
    Compute symmetric KL divergence using KDE for density estimation, avoiding infinities.
    """
    kde1 = gaussian_kde(sample1, bw_method=bandwidth)
    kde2 = gaussian_kde(sample2, bw_method=bandwidth)

    x_vals = np.linspace(min(min(sample1), min(sample2)), max(max(sample1), max(sample2)), 1000)
    p = kde1(x_vals)
    q = kde2(x_vals)

    # Avoid zero probabilities by adding a small constant (Laplace smoothing)
    p = np.maximum(p, epsilon)
    q = np.maximum(q, epsilon)

    # Normalize to ensure proper probability distributions
    p /= np.sum(p)
    q /= np.sum(q)

    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)

    return kl_pq + kl_qp

def run_simulation(rate_var, rate_mean, model_class, model_params, n_neurons, n_sources, n_groups,
                   pattern_active, length, grid_size, w_ff, queue):
    min_rate, max_rate = rate_mean - rate_var, rate_mean + rate_var

    n_samples = 180
    input_rates = sorted(np.random.randint(min_rate, max_rate+1, n_samples))

    results = {
        "rate_var": [], "x_rsync": [], "x_sc": [], "y": [],
        "ks_stat_rsync": [], "ks_p_rsync": [], "ks_stat_sc": [], "ks_p_sc": [],
        "kl_rsync": [], "kl_sc": [],
        "f1_rsync": [], "f1_sc": [],
        "f1_rsync_all": [], "f1_sc_all": [],
    }

    for iter in range(10):

        w_exc, w_som, w_pv = build_connectivity(grid_size)
        w_inh = w_pv + w_som * 0

        x_sc = []
        x_rsync = []
        y = []

        x_sc_all = []
        x_rsync_all = []

        for i, input_rate in enumerate(input_rates):
            print("VAR", rate_var, i, "/", len(input_rates))

            for fam in range(2):
                model = model_class(n_neurons=n_neurons, n_inputs=n_neurons,
                                            w_exc=w_exc, w_inh=w_inh, w_ff=w_ff, **model_params)
                if fam == 1:
                    n_max = 0
                    pattern_orig = sum(
                                [list(np.arange(i, i + 4)) for i in np.arange(n_max, int(n_max + 10 * n_groups), 10)], [])
                    pattern_cur = sorted(np.random.choice(pattern_orig, pattern_active, replace=False))
                else:
                    pattern_orig = sorted(np.random.choice(np.arange(n_neurons), pattern_active, replace=False))
                    pattern_cur = pattern_orig.copy()

                poisson_input = generate_lgn_inputs(n_neurons, n_sources, pattern_cur, input_rate, length + 1)
                _, spikes = model.simulate(length=length, external_input=poisson_input)

                x_sc.append(spikes[pattern_orig].sum(1).mean())  # x_sc.append(spikes[pattern_cur].sum(1))
                x_rsync.append(measure_rsync(spikes[pattern_orig]))
                y.append(fam)

                x_sc_all.append(spikes.sum(1).mean())  # x_sc.append(spikes[pattern_cur].sum(1))
                x_rsync_all.append(measure_rsync(spikes))

        x_rsync = np.array(x_rsync)
        x_sc = np.array(x_sc)
        y = np.array(y)
        x_rsync_all = np.array(x_rsync_all)
        x_sc_all = np.array(x_sc_all)

        mask = np.isfinite(x_rsync) & np.isfinite(x_sc) & np.isfinite(x_rsync_all) & np.isfinite(x_sc_all)
        x_rsync, x_sc, x_rsync_all, x_sc_all, y = x_rsync[mask], x_sc[mask], x_rsync_all[mask], x_sc_all[mask], y[mask]

        print("DONE", rate_var, min_rate, max_rate, len(y))

        ids_fam = np.where(y == 1)[0]
        ids_new = np.where(y == 0)[0]

        ks_stat_rsync, ks_p_rsync = kolmogorov_smirnov_test(x_rsync[ids_fam], x_rsync[ids_new])
        kl_rsync = symmetric_kl_divergence(x_rsync[ids_fam], x_rsync[ids_new])

        ks_stat_sc, ks_p_sc = kolmogorov_smirnov_test(x_sc[ids_fam], x_sc[ids_new])
        kl_sc = symmetric_kl_divergence(x_sc[ids_fam], x_sc[ids_new])

        # Familiarity classification
        skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
        f1_scores = {}

        for x_key, x_val in {"SC Input": x_sc, "Total SC": x_sc_all,
                             "Rsync Input": x_rsync, "Total Rsync": x_rsync_all}.items():

            X = np.array(x_val).reshape(-1, 1)
            f1_scores_cur = []

            for train_index, val_index in skf.split(X, y):  # Stratified splits
                # Split into training and validation sets
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]

                # Train model using F1-score
                clf = LogisticRegression()
                clf.fit(X_train, y_train)
                y_val_pred = clf.predict(X_val)
                f1 = f1_score(y_val, y_val_pred)
                f1_scores_cur.append(f1)

            f1_scores[x_key] = np.mean(f1_scores_cur)

            def best_t(x, y, n=200):
                ts = np.linspace(x.min(), x.max(), n)
                return max((f1_score(y, x >= t), t) for t in ts)

            def cv_f1(x, y, skf, n=200):
                f1s = []
                for train, val in skf.split(x, y):
                    _, t = best_t(x[train], y[train], n=n)
                    f1s.append(f1_score(y[val], x[val] >= t))
                return np.mean(f1s)

            f1_thr = cv_f1(X, y, skf)
            print(iter, "RATE VAR", rate_var, "MEASURE", x_key, "SAMPLES", len(y), "F1", np.round(f1_scores[x_key], 4), "Threshold F1", np.round(f1_thr, 4))
            f1_scores[x_key] = f1_thr

        # Send results back to main process
        results["rate_var"] = rate_var
        results["ks_stat_rsync"].append(ks_stat_rsync)
        results["ks_p_rsync"].append(ks_p_rsync)
        results["ks_stat_sc"].append(ks_stat_sc)
        results["ks_p_sc"].append(ks_p_sc)
        results["kl_rsync"].append(kl_rsync)
        results["kl_sc"].append(kl_sc)
        results["f1_rsync"].append(f1_scores["Rsync Input"])
        results["f1_sc"].append(f1_scores["SC Input"])
        results["f1_rsync_all"].append(f1_scores["Total Rsync"])
        results["f1_sc_all"].append(f1_scores["Total SC"])
        results["x_rsync"].extend(list(x_rsync))
        results["x_sc"].extend(list(x_sc))
        results["y"].extend(list(y))

    queue.put(results)

if __name__ == "__main__":
    length = 500
    model_class = NetworkLIF
    n_sources = 1

    # parameters: CONNECTIVITY
    rate_mean = 50
    w_ff_val = 4
    grid_size = 10
    n_neurons = grid_size * grid_size * 8
    pattern_size = 40
    n_groups = int(pattern_size / 4)

    # parameters: EXPERIMENT
    pattern_active_mask = 0.0

    input_size = int(pattern_size * 1.0)
    pattern_active = int(input_size * (1 - pattern_active_mask))

    model_params = params_fixed.copy()

    # build connectivity
    w_ff = build_input_one_to_one(n_neurons=n_neurons, n_inputs=n_neurons) * w_ff_val

    rate_list = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    fig, ax = plt.subplots(figsize=(20, 7), nrows=2, ncols=5)
    plot_dir = "out/v1"
    os.makedirs(plot_dir, exist_ok=True)

    manager = mp.Manager()
    queue = manager.Queue()
    processes = []
    results = {"rate_var": [],
               "ks_stat_rsync": [], "ks_p_rsync": [], "ks_stat_sc": [], "ks_p_sc": [],
               "kl_rsync": [], "kl_sc": [],
               "f1_rsync": [], "f1_sc": [], "f1_rsync_all": [], "f1_sc_all": [],

               "ks_stat_rsync_SD": [], "ks_p_rsync_SD": [], "ks_stat_sc_SD": [], "ks_p_sc_SD": [],
               "kl_rsync_SD": [], "kl_sc_SD": [],
               "f1_rsync_SD": [], "f1_sc_SD": [], "f1_rsync_all_SD": [], "f1_sc_all_SD": [],
               }
    results_dist = {}

    for rate_var in rate_list:
        p = mp.Process(target=run_simulation, args=(rate_var, rate_mean, model_class, model_params,
                                                    n_neurons, n_sources, n_groups, pattern_active,
                                                    length, grid_size, w_ff, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    while not queue.empty():
        result_p = queue.get()
        for key in results:
            if key == "rate_var":
                results[key].append(result_p[key])
            elif "_SD" not in key:
                results[key].append(np.round(np.mean(result_p[key]), 3))
                results[f"{key}_SD"].append(np.round(np.std(result_p[key]), 3))

        results_dist[result_p["rate_var"]] = (result_p["x_rsync"], result_p["x_sc"], result_p["y"])

    # Sort results
    df = pd.DataFrame(results)
    df = df.sort_values(by=["rate_var"], ascending=True)
    df.to_csv(f"{plot_dir}/results.csv", index=False)

    records = []
    for rate_var, (x_rsync, x_sc, y) in results_dist.items():
        for i in range(len(x_rsync)):  # assumes all arrays are the same length
            records.append({
                "rate_var": rate_var,
                "x_rsync": x_rsync[i],
                "x_sc": x_sc[i],
                "y": y[i]
            })
    df1 = pd.DataFrame(records)
    df1 = df1.sort_values(by=["rate_var"], ascending=True)
    df1.to_csv(f"{plot_dir}/results_dist.csv", index=False)

    print("ALL DONE, START PLOTTING")

    # Create summary plot
    fig, ax = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)

    ax[0].plot(rate_list, df["ks_stat_rsync"], color="orangered", label="Rsync", marker='o')
    ax[0].fill_between(
        rate_list,
        df["ks_stat_rsync"] - df["ks_stat_rsync_SD"],
        df["ks_stat_rsync"] + df["ks_stat_rsync_SD"],
        alpha=0.2, color="orangered"
    )
    ax[0].plot(rate_list, df["ks_stat_sc"], color="teal", label="Spike count", marker='o')
    ax[0].fill_between(
        rate_list,
        df["ks_stat_sc"] - df["ks_stat_sc_SD"],
        df["ks_stat_sc"] + df["ks_stat_sc_SD"],
        alpha=0.2, color="teal"
    )
    ax[0].set_title("KS statistic", fontsize=20, pad=15)
    ax[0].set_xlabel("Input rate variability", fontsize=18, labelpad=10)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].legend(fontsize=16)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].plot(rate_list, df["kl_rsync"], color="orangered", label="Rsync", marker='o')
    ax[1].fill_between(
        rate_list,
        df["kl_rsync"] - df["kl_rsync_SD"],
        df["kl_rsync"] + df["kl_rsync_SD"],
        alpha=0.2, color="orangered"
    )
    ax[1].plot(rate_list, df["kl_sc"], color="teal", label="Spike count", marker='o')
    ax[1].fill_between(
        rate_list,
        df["kl_sc"] - df["kl_sc_SD"],
        df["kl_sc"] + df["kl_sc_SD"],
        alpha=0.2, color="teal"
    )
    ax[1].set_title("Symm. KL-divergence", fontsize=20, pad=15)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].set_xlabel("Input rate variability", fontsize=18, labelpad=10)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/summary.png")
    fig.savefig(f"{plot_dir}/summary.svg")
    #plt.show()

    # Create F1 plot
    fig, ax = plt.subplots(figsize=(5, 8), nrows=2, ncols=1)

    ax[0].plot(rate_list, df["f1_rsync_all"], color="orangered", label="Rsync", marker="o")
    ax[0].fill_between(
        rate_list,
        df["f1_rsync_all"] - df["f1_rsync_all_SD"],
        df["f1_rsync_all"] + df["f1_rsync_all_SD"],
        alpha=0.2, color="orangered"
    )
    ax[0].plot(rate_list, df["f1_sc_all"], color="teal", label="Spike count", marker="o")
    ax[0].fill_between(
        rate_list,
        df["f1_sc_all"] - df["f1_sc_all_SD"],
        df["f1_sc_all"] + df["f1_sc_all_SD"],
        alpha=0.2, color="teal"
    )
    ax[0].set_title("All neurons", fontsize=24, pad=15)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[0].set_ylabel("F1 score", fontsize=22, labelpad=10)
    ax[0].set_xticks([])
    ax[0].set_ylim(0.3, 1.05)
    ax[0].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[1].plot(rate_list, df["f1_rsync"], color="orangered", label="Rsync", marker="o")
    ax[1].fill_between(
        rate_list,
        df["f1_rsync"] - df["f1_rsync_SD"],
        df["f1_rsync"] + df["f1_rsync_SD"],
        alpha=0.2, color="orangered"
    )
    ax[1].plot(rate_list, df["f1_sc"], color="teal", label="Spike count", marker="o")
    ax[1].fill_between(
        rate_list,
        df["f1_sc"] - df["f1_sc_SD"],
        df["f1_sc"] + df["f1_sc_SD"],
        alpha=0.2, color="teal"
    )
    ax[1].set_title("Input pattern", fontsize=24, pad=15)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].tick_params(axis='y', labelsize=20)
    ax[1].set_xlabel("Input rate variability", fontsize=22, labelpad=12)
    ax[1].set_ylabel("F1 score", fontsize=22, labelpad=10)
    ax[1].set_ylim(0.3, 1.05)
    ax[1].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_xticks([0, 10, 20, 30, 40])

    ax[1].legend(fontsize=18, loc='center left')

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/f1.png")
    fig.savefig(f"{plot_dir}/f1.svg")

    #plt.show()

    # Create distributions plot
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(figsize=(10, 6), nrows=nrows, ncols=ncols)

    for i_col, rate_var in enumerate((0, 20, 40)):
        print("plot", rate_var)
        df1_rate = df1[df1["rate_var"] == rate_var]
        df1_fam = df1_rate[df1_rate["y"] == True]
        df1_new = df1_rate[df1_rate["y"] == False]

        sns.kdeplot(df1_fam["x_rsync"], fill=True, color="orangered", alpha=0.25, linewidths=2,
                    label="Rsync: familiar", ax=ax[0, i_col])
        sns.kdeplot(df1_new["x_rsync"], fill=True, color="coral", alpha=0.15, linewidths=2, linestyles="dashed",
                    label="Rsync: new", ax=ax[0, i_col])
        ax[0, i_col].set_title(f"Variability {rate_var}", fontsize=22, pad=15)

        sns.kdeplot(df1_fam["x_sc"], fill=True, color="teal", alpha=0.25, linewidths=2,
                    label="SC: familiar", ax=ax[1, i_col])
        sns.kdeplot(df1_new["x_sc"], fill=True, color="cadetblue", alpha=0.15, linewidths=2, linestyles="dashed",
                    label="SC: new", ax=ax[1, i_col])

    for i_col in range(ncols):
        if i_col == ncols - 1:
            ax[0, i_col].legend(fontsize=18)
            ax[1, i_col].legend(fontsize=18)

        ax[0, i_col].spines['top'].set_visible(False)
        ax[0, i_col].spines['right'].set_visible(False)
        ax[0, i_col].set_ylim(0, 48)
        ax[0, i_col].set_xlim(0, 0.2)
        ax[0, i_col].tick_params(axis='x', labelsize=18)
        ax[0, i_col].tick_params(axis='y', labelsize=18)
        ax[0, i_col].set_xlabel("")
        ax[0, i_col].set_xticks([0, 0.1, 0.2])
        ax[0, i_col].set_yticks([0, 10, 20, 30, 40])

        ax[1, i_col].spines['top'].set_visible(False)
        ax[1, i_col].spines['right'].set_visible(False)
        ax[1, i_col].set_ylim(0, 0.53)
        ax[1, i_col].set_xlim(0, 43)
        ax[1, i_col].tick_params(axis='x', labelsize=18)
        ax[1, i_col].tick_params(axis='y', labelsize=18)
        ax[1, i_col].set_xlabel("")
        ax[1, i_col].set_xticks([0, 20, 40])
        ax[1, i_col].set_yticks([0, 0.1, 0.2, 0.3, 0.4])

        if i_col > 0:
            ax[0, i_col].set_yticks([])
            ax[1, i_col].set_yticks([])
            ax[0, i_col].set_ylabel("")
            ax[1, i_col].set_ylabel("")
        else:
            ax[0, i_col].set_ylabel("Density", fontsize=20, labelpad=13)
            ax[1, i_col].set_ylabel("Density", fontsize=20, labelpad=13)

    fig.tight_layout()
    fig.savefig(f"{plot_dir}/distributions.png")
    fig.savefig(f"{plot_dir}/distributions.svg")