import itertools
import os

from measure import measure_rsync, measure_mean_sc

# Experimental parameters to change
n_neurons = 1000
jitter_window = None

exp_name = f"{n_neurons}"
if jitter_window:
    exp_name = f"{n_neurons}_{jitter_window}"

data_dir = "data"
log_dir = os.path.join(data_dir, "logs")
plot_dir = os.path.join(data_dir, "plots")
param_dir = os.path.join(data_dir, "params")

params_fixed  = {
                "dt": 1.0,
                "V_thr": -55,
                "E_L": -65,
                "t_refr": 3.0,
                "tau_m": 10.0,
                }

params_change = {
                "w_exc_mean": {"min": 0.35, "max": 1.0, "step": 0.01},
                "w_som_mean": {"min": 0.0, "max": 5.0, "step": 0.1},
                "w_pv_mean": {"min": 0.0, "max": 5.0, "step": 0.1}
                }

metric_names_test = ["sc", "rsync"]

input_rate_var = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

metrics = {"sc": {"func": measure_mean_sc, "kwargs": {}},
           "rsync": {"func": measure_rsync, "kwargs": {}}
           }

if n_neurons == 1000:
    n_patterns_list = [10, 50, 100, 150]
    w_exc_p_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    w_som_p_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
elif n_neurons == 5000:
    n_patterns_list = [100, 200, 300, 400, 500]
    w_exc_p_list = [0.4,]
    w_som_p_list = [0.02,]
    
combinations = list(itertools.product(n_patterns_list, w_exc_p_list, w_som_p_list))
