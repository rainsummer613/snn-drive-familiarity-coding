import os
import numpy as np
import pandas as pd
import itertools
import re
import ast

import matplotlib.pyplot as plt

from scipy.stats import shapiro, ttest_rel, wilcoxon

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / np.std(diff, ddof=1)

def parse_log(lines, combine_ratio=True, filename=None):
    # 1) to text
    text = "".join(lines) if not isinstance(lines, str) else lines

    # 2) last 'RES ::' line
    last_res_line = None
    for line in reversed(text.splitlines()):
        if "RES ::" in line:
            last_res_line = line.strip()
            break
    if not last_res_line:
        raise ValueError(f"No 'RES ::' entry found in file: {filename}")

    # 3) extract dict payload from that line
    try:
        payload = last_res_line.split("RES ::", 1)[1].strip()
    except Exception:
        raise ValueError(f"Bad 'RES ::' line: {last_res_line}")

    payload = payload.replace("...", "")
    lb = payload.find("{")
    rb = payload.rfind("}")
    if lb == -1 or rb == -1 or rb <= lb:
        raise ValueError(f"No dict braces found on 'RES ::' line (filename={filename})")
    dict_str = payload[lb:rb+1]

    try:
        results_dict = ast.literal_eval(dict_str)
    except Exception as e:
        raise ValueError(f"Could not parse RES dict (filename={filename}): {e}")

    # 4) flatten to columns
    flattened = {}
    ratio_col = None

    for section, metrics in results_dict.items():
        # Example sections: 'f1_fam', 'f1_class'
        for metric_name, values in metrics.items():
            col_name = f"{section} {metric_name}"  # e.g. "f1_fam Rsync Input"

            # handle Ratio(s)
            if "Ratio" in metric_name:
                if combine_ratio and ratio_col is None:
                    ratio_col = values
                elif not combine_ratio:
                    flattened[col_name] = values
                continue

            if isinstance(values, list):
                flattened[col_name] = values

    df = pd.DataFrame(flattened)

    # 5) single 'Ratio' column at front
    if combine_ratio and ratio_col is not None:
        if isinstance(ratio_col, list) and len(ratio_col) == len(df):
            df.insert(0, "Ratio", ratio_col)
        else:
            # broadcast a scalar (or first element) if needed
            val = ratio_col[0] if isinstance(ratio_col, list) and ratio_col else 1.0
            df.insert(0, "Ratio", [val] * len(df))

    # 6) rename columns
    ren = {}
    for c in list(df.columns):
        if c.startswith("f1_fam "):
            ren[c] = "f1 " + c[len("f1_fam "):]
        elif c.startswith("f1_class "):
            ren[c] = "class_f1 " + c[len("f1_class "):]
    if ren:
        df.rename(columns=ren, inplace=True)

    # 7) experiment parameters for nested keys 
    n_val, exc_val, inh_val = None, None, None
    for line in text.splitlines():
        if line.startswith("EXPERIMENT PARAMETERS ::"):
            n_patterns = re.search(r"n\s([0-9]+)", line)[0][2:]
            w_exc_p    = re.search(r"exc\s([0-9.]+)", line)[0][4:]
            w_inh_p    = re.search(r"inh\s([0-9.]+)", line)[0][4:]
            break

    return n_patterns, w_exc_p, w_inh_p, df

if __name__ == "__main__":

    n_patterns_list = [100, 200, 300, 400, 500]
    w_exc_p_list = [0.4, ]
    w_som_p_list = [0.02, ]
        
    combinations = list(itertools.product(n_patterns_list, w_exc_p_list, w_som_p_list))

    log_dir = "out/amn/5000"

    results = {}
    
    for f in os.listdir(log_dir):
        if f[-3:] == "out":
    
            try:
                f_path = os.path.join(log_dir, f)
            
                with open(f_path, "r") as file:
                    lines = file.readlines()
                    f_path = os.path.join(log_dir, f) 
            
                n_patterns, w_exc_p, w_som_p, df = parse_log(lines, filename=f)
            
                combo_key = f"{w_exc_p}_{w_som_p}"
                
                if combo_key not in results:
                    results[combo_key] = {}
                results[combo_key][n_patterns] = df
                
            except Exception as e:
                print(str(e))

    group_labels = [100, 200, 300, 400, 500]
    palette = {'Rsync': '#ff8f65',  # blue
               'SC': '#70b3b3'}     # green
    edgecolor = "#3a3a3a"
    metrics = ("Class", "Fam")
    orig_metrics_rsync = ("class_f1 Max pattern Rsync", "f1 Rsync Max SC")
    orig_metrics_sc = ("class_f1 Max pattern SC", "f1 SC Max SC")
    titles = ("Input classification", "Familiarity detection")
    
    plot_dir = "data/amn/5000"
    
    for key in ["0.4_0.02", ]:
        print(key)
        combo_data = results[key]
    
        summary = []
        stats = {"Class": [], "Fam": []}
    
        fig, ax = plt.subplots(figsize=(4.25, 6.5), nrows=2, ncols=1)
        
        for n_patterns, df in combo_data.items():
            summary.append({
                "Num patterns": n_patterns,
                "Fam Rsync": df["f1 Rsync Max SC"].mean(),
                "Fam Rsync SD": df["f1 Rsync Max SC"].std(),
                "Fam SC": df["f1 SC Max SC"].mean(),
                "Fam SC SD": df["f1 SC Max SC"].std(),
                "Class SC": df["class_f1 Max pattern SC"].mean(),
                "Class SC SD": df["class_f1 Max pattern SC"].std(),
                "Class Rsync": df["class_f1 Max pattern Rsync"].mean(),
                "Class Rsync SD": df["class_f1 Max pattern Rsync"].std(),
            })
    
            for i, col in enumerate(metrics):
                rsync = df[orig_metrics_rsync[i]].values
                sc = df[orig_metrics_sc[i]].values
                diff = rsync - sc
    
                #print(col, rsync, sc, cohens_d(rsync, sc))
            
                # Shapiro-Wilk test for normality of differences
                shapiro_stat, shapiro_p = shapiro(diff)
            
                # Paired t-test
                t_stat, t_p = ttest_rel(rsync, sc)
            
                # Wilcoxon signed-rank test
                try:
                    wilcoxon_stat, wilcoxon_p = wilcoxon(rsync, sc)
                except ValueError:
                    wilcoxon_stat, wilcoxon_p = np.nan, np.nan  # if all differences are 0
            
                # Cohen's d
                d = cohens_d(rsync, sc)
            
                stats[col].append({
                    "n_patterns": n_patterns,
                    "Shapiro W": shapiro_stat,
                    "Shapiro p": shapiro_p,
                    "t-test t": t_stat,
                    "t-test p": t_p,
                    "Wilcoxon stat": wilcoxon_stat,
                    "Wilcoxon p": wilcoxon_p,
                    "Cohen's d": d
                })
            
        summary_df = pd.DataFrame(summary).sort_values(by="Num patterns")
    
        for i, col in enumerate(metrics):
            stats_df = pd.DataFrame(stats[col]).sort_values(by="n_patterns")
        
            print("STATS", col)
            print(stats_df)
            
            # Rsync line and shaded area
            ax[i].plot(summary_df["Num patterns"], summary_df[f"{col} Rsync"], color="orangered", label="Rsync", marker='o')
            ax[i].fill_between(
                    summary_df["Num patterns"],
                    summary_df[f"{col} Rsync"] - summary_df[f"{col} Rsync SD"],
                    summary_df[f"{col} Rsync"] + summary_df[f"{col} Rsync SD"],
                    alpha=0.2, color="orangered"
                )
                
            # SC line and shaded area
            ax[i].plot(summary_df["Num patterns"], summary_df[f"{col} SC"], color="teal", label="Spike count", marker='o')
            ax[i].fill_between(
                    summary_df["Num patterns"],
                    summary_df[f"{col} SC"] - summary_df[f"{col} SC SD"],
                    summary_df[f"{col} SC"] + summary_df[f"{col} SC SD"],
                    alpha=0.2, color="teal"
                )
                
            # Styling
            ax[i].set_title(titles[i], fontsize=20, pad=15)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].tick_params(axis='x', labelsize=16)
            ax[i].tick_params(axis='y', labelsize=16)
            ax[i].set_ylabel("F1 score", fontsize=18, labelpad=10)
            ax[i].set_ylim(0.3, 1.05)
    
            if i == 1:
                ax[i].axhline(y=0.5, color="black", linestyle="dashed", linewidth=1.5, label="Baseline")
                ax[i].set_xlabel("Familiar patterns", fontsize=18, labelpad=10)
                #ax[i].set_xticks([100, 200, 300, 400, 500])
            elif i == 0:
                ax[i].legend(fontsize=16, loc='lower left')
                ax[i].set_xticks([])
            
        fig.tight_layout(pad=2.5)
        fig.savefig(f"{plot_dir}/summary.png", dpi=500)
        fig.savefig(f"{plot_dir}/summary.svg")
        plt.show()
    
