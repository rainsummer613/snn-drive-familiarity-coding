import os
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    log_dir  = "out/v1"
    plot_dir = "data/v1"
    
    df = pd.read_csv(os.path.join(plot_dir, "results.csv"))
    rate_list = df["rate_var"].to_list()
    
    df1 = pd.read_csv(os.path.join(plot_dir, "results_dist.csv"))

    # Create F1 plot
    fig, ax = plt.subplots(figsize=(4.0, 6.0), nrows=2, ncols=1)
    
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
    ax[0].set_title("All neurons", fontsize=20, pad=15)
    ax[0].tick_params(axis='x', labelsize=16)
    ax[0].tick_params(axis='y', labelsize=16)
    ax[0].set_ylabel("F1 score", fontsize=18, labelpad=10)
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
    ax[1].set_title("Input pattern", fontsize=20, pad=15)
    ax[1].tick_params(axis='x', labelsize=16)
    ax[1].tick_params(axis='y', labelsize=16)
    ax[1].set_xlabel("Input rate variability", fontsize=18, labelpad=12)
    ax[1].set_ylabel("F1 score", fontsize=18, labelpad=10)
    ax[1].set_ylim(0.3, 1.05)
    ax[1].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5)  #, label="Baseline")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].set_xticks([0, 10, 20, 30, 40])
    
    ax[1].legend(fontsize=18, loc='center left')
    
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/f1.png")
    fig.savefig(f"{plot_dir}/f1.svg")
    
    plt.show()

    # Create distributions plot
    nrows, ncols = 2, 3
    fig, ax = plt.subplots(figsize=(10, 6), nrows=nrows, ncols=ncols)
    
    for i_col, rate_var in enumerate((0, 20, 40)):
        print("plot", rate_var)
        df1_rate = df1[df1["rate_var"] == rate_var]
        df1_fam = df1_rate[df1_rate["y"] == True]
        df1_new = df1_rate[df1_rate["y"] == False]
    
        sns.kdeplot(df1_fam["x_rsync"], fill=True, color="orangered", alpha=0.25, linewidths=2,
                                label="familiar", ax=ax[0, i_col])
        sns.kdeplot(df1_new["x_rsync"], fill=True, color="coral", alpha=0.15, linewidths=2, linestyles="dashed",
                                label="new", ax=ax[0, i_col])
        ax[0, i_col].set_title(f"Variability {rate_var}", fontsize=22, pad=15)
    
        sns.kdeplot(df1_fam["x_sc"], fill=True, color="teal", alpha=0.25, linewidths=2,
                                label="familiar", ax=ax[1, i_col])
        sns.kdeplot(df1_new["x_sc"], fill=True, color="cadetblue", alpha=0.15, linewidths=2, linestyles="dashed",
                                label="new", ax=ax[1, i_col])
    
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
    