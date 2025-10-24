import os
import numpy as np
import pandas as pd
import re
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmcrameri.cm as cmc
import seaborn as sns

plot_dir = "plots/abstract_multiclass"
root_dir = "data/abstract_multiclass"
target_subdirs = ["n_10", "n_50", "n_100", "n_150"]

color_95 = "#0B3C3C"  # "#725f74"
color_50 = "#B83301"  # "#9f6268"  

color_rsync = "orangered"  # "#d26b5c"
color_sc    = "teal"  # "#2d4d72"

pad, labelpad = 15, 12
tick_fontsize = 20
legend_fontsize = 22
label_fontsize = 22
title_fontsize = 26

if __name__ == "__main__":

    # INDIVIDUAL PLOTS (PER PARAMETER CONFIGURATION)
    
    for root, dirs, files in os.walk(root_dir):
    
        if "0.7_0.03_0.0\\size_100\\n_100" in root or "0.7_0.03_0.0\\size_100\\n_150" in root \
        or "0.6_0.03_0.0\\size_100\\n_50" in root or "0.7_0.07_0.0\\size_100\\n_10" in root:
        #if "size_100" in root and any(subdir in root for subdir in target_subdirs):
    
            n_patterns = re.search(r"n_(\d+)", root).group(1)
        
            completion_file_path = os.path.join(root, "completion.csv")
            fam_file_path = os.path.join(root, "fam F1 score.csv")
            class_file_path = os.path.join(root, "class.csv")
        
            df_completion = pd.read_csv(completion_file_path)
            df_fam = pd.read_csv(fam_file_path)
            df_class = pd.read_csv(class_file_path)
            
            print(root)
        
            fig, ax = plt.subplots(figsize=(5, 12), nrows=3, ncols=1)
        
            # PLOT Pattern classification
    
            ax[1].plot(df_class["Ratio"], df_class["Max pattern Rsync"], color=color_rsync, label="Rsync")
            ax[1].fill_between(
                        df_class["Ratio"],
                        df_class["Max pattern Rsync"] - df_class["Max pattern Rsync SD"],
                        df_class["Max pattern Rsync"] + df_class["Max pattern Rsync SD"],
                        alpha=0.25, color=color_rsync,
                    )
            ax[1].plot(df_class["Ratio"], df_class["Max pattern SC"], color=color_sc, label="SC")
            ax[1].fill_between(
                        df_class["Ratio"],
                        df_class["Max pattern SC"] - df_class["Max pattern SC SD"],
                        df_class["Max pattern SC"] + df_class["Max pattern SC SD"],
                        alpha=0.25, color=color_sc,
                    )
                
            ax[0].set_title(f"{n_patterns} patterns", fontsize=title_fontsize, pad=pad)
            #ax[0].set_xlabel("Intra-pattern connections", fontsize=16)
            ax[1].set_ylabel("Macro F1 score", fontsize=label_fontsize, labelpad=labelpad)
                
            # PLOT Familiarity detection
                
            ax[0].plot(df_fam["Ratio"], df_fam["Rsync Max SC"], color=color_rsync, label="Rsync")
            ax[0].fill_between(
                        df_fam["Ratio"],
                        df_fam["Rsync Max SC"] - df_fam["Rsync Max SC SD"],
                        df_fam["Rsync Max SC"] + df_fam["Rsync Max SC SD"],
                        alpha=0.25, color=color_rsync,
                    )
            ax[0].plot(df_fam["Ratio"], df_fam["SC Max SC"], color=color_sc, label="SC")
            ax[0].fill_between(
                        df_fam["Ratio"],
                        df_fam["SC Max SC"] - df_fam["SC Max SC SD"],
                        df_fam["SC Max SC"] + df_fam["SC Max SC SD"],
                        alpha=0.25, color=color_sc,
                    )
            ax[0].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
            #ax[1].set_title("Familiarity detection", fontsize=title_fontsize, pad=pad)
            ax[0].set_xlabel("")
            ax[0].set_ylabel("F1 score", fontsize=label_fontsize, labelpad=labelpad)
            ax[0].legend(fontsize=legend_fontsize)
        
            # PLOT Completion
                
            ax[2].plot(df_completion["Ratio"], df_completion["Above 95%"], color=color_95, label="Above 95%")
            ax[2].fill_between(
                        df_completion["Ratio"],
                        df_completion["Above 95%"] - df_completion["Above 95% SD"],
                        df_completion["Above 95%"] + df_completion["Above 95% SD"],
                        alpha=0.25, color=color_95,
                    )
            ax[2].plot(df_completion["Ratio"], df_completion["Above 50%"], color=color_50, label="Above 50%")
            ax[2].fill_between(
                        df_completion["Ratio"],
                        df_completion["Above 50%"] - df_completion["Above 50% SD"],
                        df_completion["Above 50%"] + df_completion["Above 50% SD"],
                        alpha=0.25, color=color_50,
                    )
            #ax[2].set_title("Pattern completion", fontsize=title_fontsize, pad=pad)
            ax[2].set_xlabel("Input proportion", fontsize=label_fontsize, labelpad=labelpad)
            ax[2].set_ylabel("Proportion of completed", fontsize=label_fontsize, labelpad=labelpad)
            ax[2].legend(fontsize=legend_fontsize)
        
        
            for i in range(3):
                ax[i].spines['top'].set_visible(False)
                ax[i].spines['right'].set_visible(False)
                ax[i].tick_params(axis='x', labelsize=tick_fontsize)
                ax[i].set_xticks((0.6, 0.7, 0.8, 0.9, 1.0))
                ax[i].tick_params(axis='y', labelsize=tick_fontsize)
                ax[i].set_ylim(0.0, 1.05)
            
            fig.tight_layout()
            #plt.show()
            for ext in ("svg", "png"):
                fig_path = os.path.join(root, f"summary.{ext}")
                fig.savefig(fig_path, dpi=300)
            plt.close()

    target_file = "fam F1 score.csv"
    target_columns = ["SC Max SC", "Rsync Max SC"]
    #target_columns = ["SC Total", "Rsync Total"]
    
    pad, labelpad = 16, 15
    tick_fontsize = 20
    legend_fontsize = 24
    label_fontsize = 24
    title_fontsize = 28
    
    # Initialize results dictionary
    results_fam = {col: {n: [] for n in target_subdirs} for col in target_columns}
    
    # Gather data
    for folder_name in os.listdir(root_dir):
        match = re.match(r"([\d.]+)_([\d.]+)", folder_name)
        if match:
            x_val, y_val = float(match[1]), float(match[2])
            base_path = os.path.join(root_dir, folder_name, "size_100")
    
            for n_dir in target_subdirs:
                file_path = os.path.join(base_path, n_dir, target_file)
                if not os.path.isfile(file_path):
                    continue
                df = pd.read_csv(file_path)
                ratio_row = df[df["Ratio"] == 1.0]
                if ratio_row.empty:
                    continue
    
                for col in target_columns:
                    val = ratio_row[col].values[0]
                    results_fam[col][n_dir].append((x_val, y_val, val))
    
    # Plotting
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(target_subdirs),
        figsize=(5 * len(target_subdirs), 10),
        sharey=True
    )
    
    vmin, vmax = 0.49, 1.0
    cbar_axes = []
    
    # First row: SC Max SC
    for i, n_dir in enumerate(target_subdirs):
        ax = axes[0, i]
        df_heat = pd.DataFrame(results_fam[target_columns[0]][n_dir], columns=["x", "y", "F1"])
        heat_data = df_heat.pivot(index="y", columns="x", values="F1").sort_index(ascending=False)
        sns_plot = sns.heatmap(
            heat_data, ax=ax, 
            cmap=cmc.lipari, vmin=vmin, vmax=vmax,
            cbar=False,
            # annot=True, fmt=".2f", annot_kws={'size': 11}
        )
        #if i == len(target_subdirs) - 1:
        #    cbar_axes.append(sns_plot.collections[0].colorbar)
        title = n_dir.split("_")[-1]
        ax.set_title(f"{title} patterns", fontsize=title_fontsize, pad=pad)
        if i == 0:
            ax.set_ylabel("Inhibition cross-pattern", fontsize=label_fontsize, labelpad=labelpad)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, right=False)
        ax.set_xlabel("")
    
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
    
        ax.set_xticks([])
        ax.tick_params(labelsize=tick_fontsize)
    
    # Second row: Rsync Max SC
    for i, n_dir in enumerate(target_subdirs):
        ax = axes[1, i]
        df_heat = pd.DataFrame(results_fam[target_columns[1]][n_dir], columns=["x", "y", "F1"])
        heat_data = df_heat.pivot(index="y", columns="x", values="F1").sort_index(ascending=False)
        sns_plot = sns.heatmap(
            heat_data, ax=ax,
            cmap=cmc.lipari, vmin=vmin, vmax=vmax,
            cbar=False,
            # annot=True, fmt=".2f", annot_kws={'size': 11}, annot_kws={'size': 11}
        )
    
        if i == 0:
            ax.set_ylabel("Inhibition cross-pattern", fontsize=label_fontsize, labelpad=labelpad)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, right=False)
    
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        
        ax.tick_params(labelsize=tick_fontsize)
        ax.set_xlabel("Excitation in-pattern ", fontsize=label_fontsize, labelpad=labelpad)
    
    fig.text(0.01, 0.73, "Spike count", fontsize=title_fontsize, rotation=90, va='center')
    fig.text(0.01, 0.32, "Rsync", fontsize=title_fontsize, rotation=90, va='center')
    
    plt.tight_layout(pad=2.0, rect=[0.03, 0, 1, 1])
    plt.show()
    
    fig.savefig(os.path.join(plot_dir, "Fam global.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, "svg/Fam global.svg"))

    results_fam_best = {"Rsync Max SC": {}, "SC Max SC": {}}

    for metric in ("Rsync Max SC", "SC Max SC"):
        for condition in results_fam[metric]:
            arr = sorted(results_fam[metric][condition], key=lambda x: x[2], reverse=True)[:20]
            arr = [(el[0], el[1]) for el in arr]
            results_fam_best[metric][condition] = arr
    
    r = results_fam_best['Rsync Max SC']

    # Colorbar
    # Create a ScalarMappable object
    sm = cm.ScalarMappable(cmap=cmc.lipari, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    # Create a figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(1, 5))  # Adjust the size to your liking
    ax.set_axis_off()
    
    # Add the colorbar to the axis
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.outline.set_visible(False)
    
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(14)
    
    fig.savefig(os.path.join(plot_dir, f"F1_cbar.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, f"svg/F1_cbar.svg"))

    threshold = 0.75
    results_thr_rsync = {}
    
    for group, entries in results_fam['Rsync Max SC'].items():
        filtered_pairs = [(x, y) for (x, y, val) in entries if val > threshold]
        results_thr_rsync[group] = filtered_pairs

    results_thr_sc = {}

    for group, entries in results_fam['SC Max SC'].items():
        filtered_pairs = [(x, y) for (x, y, val) in entries if val > threshold]
        results_thr_sc[group] = filtered_pairs

    # OUTPUT RATE EFFICIENCY
    target_file = "stats.csv"
    target_column = "SC Fam"
    
    # Store extracted values
    results_rate = {n: [] for n in target_subdirs}
    all_values_rate = []
    
    # Collect data
    for folder_name in os.listdir(root_dir):
        match = re.match(r"([\d.]+)_([\d.]+)_0.0", folder_name)
        if match:
            x_val = float(match[1])
            y_val = float(match[2])
            base_path = os.path.join(root_dir, folder_name, "size_100")
    
            for n_dir in target_subdirs:
                file_path = os.path.join(base_path, n_dir, target_file)
                if not os.path.isfile(file_path):
                    continue
                try:
                    df = pd.read_csv(file_path)
                    ratio_row = df[df["Ratio"] == 1.0]
                    if not ratio_row.empty:
                        val = ratio_row[target_column].values[0]
                        results_rate[n_dir].append((x_val, y_val, val))
                        all_values_rate.append(val)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Determine global min/max
    vmin_rate, vmax_rate = min(all_values_rate), max(all_values_rate)
    
    # Plotting
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(target_subdirs),
        figsize=(5 * len(target_subdirs), 6.0),
        sharey=True
    )
    
    for i, n_dir in enumerate(target_subdirs):
        ax = axes[i]
        df_heat = pd.DataFrame(results_rate[n_dir], columns=["x", "y", "value"])
        df_heat["value"] = df_heat["value"].round().astype(int) * 2 
        heat_data = df_heat.pivot(index="y", columns="x", values="value").sort_index(ascending=False)
    
        sns.heatmap(
            heat_data, ax=ax, 
            cmap=cmc.lipari, vmin=vmin_rate, vmax=vmax_rate, cbar=False,
            #annot=True, fmt=".2f", annot_kws={'size': 11}, annot_kws={'size': 11}
        )
    
        # Titles and formatting
        title = n_dir.split("_")[-1]
        ax.set_title(f"{title} patterns", fontsize=title_fontsize, pad=pad)
        ax.set_xlabel("Excitation in-pattern", fontsize=label_fontsize, labelpad=labelpad)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        ax.tick_params(labelsize=tick_fontsize)
    
        if i == 0:
            ax.set_ylabel("Inhibition cross-pattern", fontsize=label_fontsize, labelpad=labelpad)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, right=False)
    
    plt.tight_layout(pad=2.0, rect=[0.03, 0, 1, 1])
    plt.show()
    
    fig.savefig(os.path.join(plot_dir, f"Rate.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, f"svg/Rate.svg"))

    # Colorbar
    # Create a ScalarMappable object
    sm = cm.ScalarMappable(cmap=cmc.lipari, norm=plt.Normalize(vmin=vmin_rate, vmax=vmax_rate))
    
    # Create a figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(1, 5))  # Adjust the size to your liking
    ax.set_axis_off()
    
    # Add the colorbar to the axis
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.outline.set_visible(False)
    
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(14)
    
    fig.savefig(os.path.join(plot_dir, f"Rate_cbar.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, f"svg/Rate_cbar.svg"))

    # OUTPUT RATE DISTRIBUTIONS
    rate_dist_rsync = {}

    for group in results_thr_rsync:
        wanted_pairs = set(results_thr_rsync[group])
        group_result = [
            val for (x, y, val) in results_rate[group]
            if (x, y) in wanted_pairs
        ]
        rate_dist_rsync[group] = group_result
    
    for k, v in rate_dist_rsync.items():
        print(k, len(v), np.mean(v))
    
    rate_dist_sc = {}
    
    for group in results_thr_sc:
        wanted_pairs = set(results_thr_sc[group])
        group_result = [
            val for (x, y, val) in results_rate[group]
            if (x, y) in wanted_pairs
        ]
        rate_dist_sc[group] = group_result

    rate_dist = {}

    for metric in results_fam_best:
        print(metric)
        rate_dist[metric] = {}
        for group in results_fam_best[metric]:
            wanted_pairs = set(results_fam_best[metric][group])
            group_result = [
                val for (x, y, val) in results_rate[group]
                if (x, y) in wanted_pairs
            ]
            rate_dist[metric][group] = group_result
    
    n_patterns = [10, 50, 100, 150]
    fig, ax = plt.subplots(figsize=(20, 4.5), nrows=1, ncols=len(n_patterns))
    
    for i, n in enumerate(n_patterns):
        ax[i].set_title(f"{n} patterns", fontsize=title_fontsize, pad=pad)
        
        sns.kdeplot(rate_dist_rsync[f"n_{n}"], fill=True, color=color_rsync, alpha=0.25, linewidths=2,
                                label="Rsync", ax=ax[i])
        sns.kdeplot(rate_dist_sc[f"n_{n}"], fill=True, color=color_sc, alpha=0.15, linewidths=2,
                                label="SC", ax=ax[i])
    
        ax[i].tick_params(axis='x', labelsize=tick_fontsize)
        ax[i].tick_params(axis='y', labelsize=tick_fontsize)
        ax[i].set_ylabel(" ")
        ax[i].set_xlabel("Output firing rate", fontsize=label_fontsize, labelpad=labelpad)
    
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    
        ax[i].set_xlim(-10, 100)  # 70
        ax[i].set_xticks((0, 25, 50, 75, 100))
        #ax[i].set_ylim(0, 0.37)  # 0.155
    
        if i > 0:
            ax[i].tick_params(axis='y', left=False, right=False)
            #ax[i].set_yticks([])
        if len(rate_dist_sc[f"n_{n}"]) > 0:
            stat, p = mannwhitneyu(rate_dist_rsync[f"n_{n}"], rate_dist_sc[f"n_{n}"], alternative='less')
            print(stat, p)
    
        #ax[i].set_title(f"{n} patterns", fontsize=20)
    
    ax[1].legend(fontsize=legend_fontsize)
    ax[0].set_ylabel("Density", fontsize=label_fontsize, labelpad=labelpad)
    
    fig.tight_layout()
    plt.show()
    
    fig.savefig(os.path.join(plot_dir, f"Rate dist.png"))
    fig.savefig(os.path.join(plot_dir, f"svg/Rate dist.svg"))

    # PATTERN CLASSIFICATION
    target_file = "class.csv"
    target_columns = ["Max pattern Rsync", "Max pattern SC"]
    
    # Initialize results dictionary
    results_class = {col: {n: [] for n in target_subdirs} for col in target_columns}
    
    # Gather data
    for folder_name in os.listdir(root_dir):
        match = re.match(r"([\d.]+)_([\d.]+)", folder_name)
        if match:
            x_val, y_val = round(10 * float(match[1]), 1), float(match[2])
            base_path = os.path.join(root_dir, folder_name, "size_100")
    
            for n_dir in target_subdirs:
                file_path = os.path.join(base_path, n_dir, target_file)
                if not os.path.isfile(file_path):
                    continue
                df = pd.read_csv(file_path)
                ratio_row = df[df["Ratio"] == 1.0]
                if ratio_row.empty:
                    continue
    
                for col in target_columns:
                    val = ratio_row[col].values[0]
                    results_class[col][n_dir].append((x_val, y_val, val))
    
    # Plotting
    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(target_subdirs),
        figsize=(5 * len(target_subdirs), 10),
        sharey=True
    )
    
    vmin, vmax = 0.0, 1.0
    cbar_axes = []
    
    # First row: SC Max SC
    for i, n_dir in enumerate(target_subdirs):
        ax = axes[0, i]
        df_heat = pd.DataFrame(results_class["Max pattern SC"][n_dir], columns=["x", "y", "F1"])
        heat_data = df_heat.pivot(index="y", columns="x", values="F1").sort_index(ascending=False)
        sns_plot = sns.heatmap(
            heat_data, ax=ax, 
            cmap=cmc.lipari, vmin=vmin, vmax=vmax,
            cbar=False,
            # annot=True, fmt=".2f", annot_kws={'size': 11}
        )
        #if i == len(target_subdirs) - 1:
        #    cbar_axes.append(sns_plot.collections[0].colorbar)
        title = n_dir.split("_")[-1]
        ax.set_title(f"{title} patterns", fontsize=24, pad=15)
        if i == 0:
            ax.set_ylabel("Inhibition cross-pattern", fontsize=22, labelpad=12)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, right=False)
        ax.set_xlabel("")
    
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
    
        ax.set_xticks([])
        ax.tick_params(labelsize=18)
    
    # Second row: Rsync Max SC
    for i, n_dir in enumerate(target_subdirs):
        ax = axes[1, i]
        df_heat = pd.DataFrame(results_class["Max pattern Rsync"][n_dir], columns=["x", "y", "F1"])
        heat_data = df_heat.pivot(index="y", columns="x", values="F1").sort_index(ascending=False)
        sns_plot = sns.heatmap(
            heat_data, ax=ax,
            cmap=cmc.lipari, vmin=vmin, vmax=vmax,
            cbar=False,
            # annot=True, fmt=".2f", annot_kws={'size': 11}, annot_kws={'size': 11}
        )
    
        if i == 0:
            ax.set_ylabel("Inhibition cross-pattern", fontsize=22, labelpad=12)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, right=False)
    
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Excitation in-pattern ", fontsize=22, labelpad=12)
    
    fig.text(0.01, 0.73, "Spike count", fontsize=24, rotation=90, va='center')
    fig.text(0.01, 0.32, "Rsync", fontsize=24, rotation=90, va='center')
    
    plt.tight_layout(pad=2.0, rect=[0.03, 0, 1, 1])
    plt.show()
    
    fig.savefig(os.path.join(plot_dir, "Class.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, "svg/Class.svg"))

    # Create a ScalarMappable object
    sm = cm.ScalarMappable(cmap=cmc.lipari, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
    # Create a figure and axis for the colorbar
    fig, ax = plt.subplots(figsize=(1, 5))  # Adjust the size to your liking
    ax.set_axis_off()
    
    # Add the colorbar to the axis
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.outline.set_visible(False)
    
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontsize(14)
    
    fig.savefig(os.path.join(plot_dir, f"Classification_cbar.png"), dpi=500)
    fig.savefig(os.path.join(plot_dir, f"svg/Classification_cbar.svg"))