import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import simpy
import scipy.stats as stats
import numpy as np
from QueueSimulation import QueueSimulation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random 
from tqdm import tqdm


def run_simulation(num_runs, num_servers, arrival_rate, service_rate, service_dist='M', max_time=1e8):
    """Run multiple simulation replications."""
    results = {
        'waiting_times_runs': [],
        'service_times_runs': [],
        'utilization_runs': [],
        'samples_runs' : [],
        'queue_length' : [],
        'waiting_times_list': []
    }

    for _ in tqdm(range(num_runs), desc=f"Simulation for {service_dist}/{service_dist}/{num_servers}"):
        # Set up environment
        env = simpy.Environment()
        sim = QueueSimulation(env, num_servers, arrival_rate, 
                                service_rate, service_dist, max_time)
        env.run(until=max_time)
        
        result = sim.get_statistics()
        results['waiting_times_runs'].append(result['avg_waiting_time'])
        results['service_times_runs'].append(result['avg_service_time'])
        results['utilization_runs'].append(result['utilization'])
        results['samples_runs'].append(result['num_samples'])
        results['waiting_times_list'].append(result['waiting_times'])
        results['queue_length'].append(result['queue_length'])
    
    return results

def compare_queue_configurations(results):
    """
    Create comprehensive and visually engaging comparison plots.
    
    Args:
        results: Dictionary of simulation results
    """
    # Prepare data
    configs = list(results.keys())
    mean_waiting_times = [results[c]['mean_waiting_time'] for c in configs]
    mean_utilizations = [results[c]['mean_utilization'] for c in configs]
    confidence_intervals = [results[c]['confidence_interval'] for c in configs]
    
    # Create figure with modified layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    avg_queue_lengths = [np.mean([np.mean(run) for run in results[c]['queue_length']]) for c in configs]


    # Box plot for waiting times
    axs[0, 0].boxplot([results[c]['waiting_times_runs'] for c in configs], labels=configs)
    axs[0, 0].set_title('Waiting Times Distribution')
    axs[0, 0].set_ylabel('Waiting Time')
    
    # Scatter plot with error bars for mean waiting times
    axs[0, 1].errorbar(configs, mean_waiting_times, 
                       yerr=[(ci[1]-ci[0])/2 for ci in confidence_intervals], 
                       fmt='o', capsize=5)
    axs[0, 1].set_title('Mean Waiting Times with Confidence Intervals')
    axs[0, 1].set_ylabel('Average Waiting Time')
    axs[0, 1].set_xticklabels(configs, rotation=45)
    
    # Scatter plot for average queue length
    axs[1, 0].scatter(configs, avg_queue_lengths)
    axs[1, 0].set_title('Average Queue Length')
    axs[1, 0].set_ylabel('Queue Length')
    axs[1, 0].set_xticklabels(configs, rotation=45)

    # Violin plot for utilization
    axs[1, 1].violinplot([results[c]['utilization_runs'] for c in configs], 
                         positions=range(len(configs)), 
                         showmeans=True)
    axs[1, 1].set_title('Utilization Distribution')
    axs[1, 1].set_xticks(range(len(configs)))
    axs[1, 1].set_xticklabels(configs, rotation=45)
    axs[1, 1].set_ylabel('Utilization')
    
    plt.tight_layout()
    plt.savefig('advanced_queue_configuration_comparison.pdf')
    
    return fig

def plot_queue_metrics(waiting_times_list, queue_length_list, config_name, rho, n_servers):
    """
    Create time series plots of average waiting times and queue length with confidence intervals.
    
    Args:
        waiting_times_list: List of waiting times lists from multiple runs
        queue_length_list: List of queue length lists from multiple runs
        config_name: Name of the configuration for file naming
        rho: System load parameter for theoretical waiting time calculation
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Ensure consistent minimum length across runs
    min_length_waiting = min([len(run) for run in waiting_times_list])
    min_length_queue = min([len(run) for run in queue_length_list])
    
    # Prepare waiting times data
    timings = [avg_waiting_times[:min_length_waiting] for avg_waiting_times in waiting_times_list]
    avg_waiting_times = np.mean(timings, axis=0)
    std_waiting_times = np.std(timings, axis=0)
    conf_interval_waiting = 1.96 * std_waiting_times / np.sqrt(len(waiting_times_list))
    
    # Prepare queue length data
    queue_lengths = [avg_queue_lengths[:min_length_queue] for avg_queue_lengths in queue_length_list]
    avg_queue_lengths = np.mean(queue_lengths, axis=0)
    std_queue_lengths = np.std(queue_lengths, axis=0)
    conf_interval_queue = 1.96 * std_queue_lengths / np.sqrt(len(queue_length_list))
    
    # Plot waiting times
    ax1.plot(avg_waiting_times, alpha=1, linewidth=2, label='Average waiting times')
    ax1.plot(np.ones(len(avg_waiting_times)) * rho/(n_servers *(1-rho)), 
             alpha=0.8, linestyle='--', 
             label=f'Theoretical waiting time (ρ/(1-ρ), ρ={rho})')
    ax1.fill_between(range(len(avg_waiting_times)), 
                     avg_waiting_times - conf_interval_waiting, 
                     avg_waiting_times + conf_interval_waiting, 
                     alpha=0.3)
    ax1.set_title('Average Waiting Times Over Time')
    ax1.set_xlabel('Customer Index')
    ax1.set_ylabel('Waiting Time')
    ax1.legend(prop = { "size": 13 })
    
    # Plot queue length
    ax2.plot(avg_queue_lengths, alpha=1, label='Average queue length')
    ax2.fill_between(range(len(avg_queue_lengths)), 
                     avg_queue_lengths - conf_interval_queue, 
                     avg_queue_lengths + conf_interval_queue, 
                     alpha=0.2)
    ax2.set_title('Average Queue Length Over Time')
    ax2.set_xlabel('Customer Index')
    ax2.set_ylabel('Queue Length')
    ax2.legend(prop = { "size": 13 })
    
    plt.tight_layout()
    plt.savefig(f'images/{config_name}_metrics.pdf')
    plt.close()

def compare_server_numbers():
    rho = 0.9  # System load
    server_capacity = 1.0 # In the assignment this is called server capacity. Why?
    n_runs = 1000
    
    #(num_servers, arrival_rate, distribution)
    configs = [
        (1, rho * server_capacity, 'M'),  # M/M/1
        (2, 2 * rho * server_capacity, 'M'),  # M/M/2
        (4, 4 * rho * server_capacity, 'M'),  # M/M/4
    ]

    results = {}
    for servers_number, lambda_, dist in configs:
        key = f"M/{dist}/{servers_number}"
        results[key] = run_simulation(n_runs, servers_number, lambda_, server_capacity, dist, n_runs)

    statistics = {}
    for config, data in results.items():
        mean_waiting_time = np.mean(data['waiting_times_runs'])
        variance_waiting_time = np.var(data['waiting_times_runs'])
        mean_service_time = np.mean(data['service_times_runs'])
        variance_service_time = np.var(data['service_times_runs'])
        mean_utilization = np.mean(data['utilization_runs'])
        variance_utilization = np.var(data['utilization_runs'])
        num_samples = np.average(data['samples_runs'])

        confidence_interval = stats.t.interval(
            0.95, 
            len(data['waiting_times_runs'])-1,
            loc=np.mean(data['waiting_times_runs']),
            scale=stats.sem(data['waiting_times_runs'])
        )
        
        statistics[config] = {
            'mean_waiting_time': mean_waiting_time,
            'variance_waiting_time': variance_waiting_time,
            'mean_utilization': mean_utilization,
            'variance_utilization': variance_utilization,
            'variance_service_time': variance_service_time,
            'confidence_interval': confidence_interval,
            'mean_num_samples': num_samples,
            'waiting_times_runs': data['waiting_times_runs'],
            'utilization_runs': data['utilization_runs'],
            'queue_length': data['queue_length']
        }

    compare_queue_configurations(statistics)

    plot_queue_metrics(
        results['M/M/1']['waiting_times_list'], 
        results['M/M/1']['queue_length'], 
        'MM1', 
        rho,
        1
    )
    plot_queue_metrics(
        results['M/M/2']['waiting_times_list'], 
        results['M/M/2']['queue_length'], 
        'MM2', 
        rho,
        2
    )
    plot_queue_metrics(
        results['M/M/4']['waiting_times_list'], 
        results['M/M/4']['queue_length'], 
        'MM4', 
        rho,
        4
    )


if __name__ == "__main__":
    params = {'legend.fontsize': 'x-large',
            #'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            }
    
    font = {'size': 12}

    plt.rc('font', **font)
    plt.rcParams.update(params)

    compare_server_numbers()
