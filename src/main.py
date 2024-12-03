import simpy
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from QueueSimulation import QueueSimulation, PriorityQueueSimulation
from utilities import *

FONT_SIZE = 17

def main():
    service_rate = 1 
    n_runs = 25
    max_time = 2e3
    rho_range = [0.8, 0.85, 0.9, 0.95]
    service_time_dists = ['M', 'D', 'H']

    rho_results = {}
    indexes = {}
    for rho in rho_range:
        rho_results[rho] = compare_server_numbers(QueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time)
        rho_results[rho].update(compare_server_numbers(PriorityQueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time))
        rho_results[rho].update(compare_server_numbers(QueueSimulation, rho, service_rate, 'D', n_runs, max_time=max_time))
        rho_results[rho].update(compare_server_numbers(QueueSimulation, rho, service_rate, 'H', n_runs, max_time=max_time))
        
        indexes.update(analyze_conf_int('M', rho_results[rho], rho))
        indexes.update(analyze_conf_int('D', rho_results[rho], rho))
        indexes.update(analyze_conf_int('H', rho_results[rho], rho))

    plot_minimum_runs(indexes, rho_range, 'M')
    plot_minimum_runs(indexes, rho_range, 'D')
    plot_minimum_runs(indexes, rho_range, 'H')

    comparisons = {}
    for dist in service_time_dists:
        for i1, rho1 in enumerate(rho_range):
            t_tests = perform_welch_t_test(dist, n_runs, rho_results[rho1], rho1)

            comparisons.update(t_tests)
    
    create_advanced_visualizations(comparisons)
    create_configuration_heatmap(rho_results)
    
    compare_queue_configurations(QueueSimulation, rho_results[0.9])

    return 0

if __name__ == "__main__":
    
    params = {'legend.fontsize': 'x-large',
            'axes.labelsize': 'x-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large' }
    
    font = {'size': FONT_SIZE}

    plt.rcParams.update(params)
    plt.rc('font', **font)

    main()

    