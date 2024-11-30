import simpy
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt

from QueueSimulation import QueueSimulation, PriorityQueueSimulation

FONT_SIZE = 15

def run_simulation(queue_method, num_runs, num_servers, arrival_rate, service_rate, max_time, service_dist='M'):
    """Run multiple simulation replications."""

    results = {
        'waiting_times_runs': [],
        'service_times_runs': [],
        'utilization_runs': [],
        'samples_runs': [],
        'queue_length': [],
        'waiting_times_list': []
    }

    for _ in tqdm(range(num_runs), desc=f"Simulation for M/{service_dist}/{num_servers}"):
        # Set up environment
        env = simpy.Environment()
        sim = queue_method(env, num_servers, arrival_rate,
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

def plot_queue_metrics(queue_method, waiting_times_list, queue_length_list, config_name, rho, n_servers):
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
    timings = [avg_waiting_times[:min_length_waiting]
        for avg_waiting_times in waiting_times_list]
    avg_waiting_times = np.mean(timings, axis=0)
    std_waiting_times = np.std(timings, axis=0)
    conf_interval_waiting = 1.96 * std_waiting_times / \
        np.sqrt(len(waiting_times_list))

    # Prepare queue length data
    queue_lengths = [avg_queue_lengths[:min_length_queue]
        for avg_queue_lengths in queue_length_list]
    avg_queue_lengths = np.mean(queue_lengths, axis=0)
    std_queue_lengths = np.std(queue_lengths, axis=0)
    conf_interval_queue = 1.96 * std_queue_lengths / \
        np.sqrt(len(queue_length_list))

    # Plot waiting times
    ax1.plot(avg_waiting_times, alpha=1, linewidth=2,
             label='Average waiting times')
    ax1.plot(np.ones(len(avg_waiting_times)) * rho/(n_servers * (1-rho)),
             alpha=0.8, linestyle='--',
             label=f'Theoretical waiting time (ρ/(1-ρ), ρ={rho})')
    ax1.fill_between(range(len(avg_waiting_times)),
                     avg_waiting_times - conf_interval_waiting,
                     avg_waiting_times + conf_interval_waiting,
                     alpha=0.3)
    ax1.set_title('Average Waiting Times Over Time')
    ax1.set_xlabel('Customer Index')
    ax1.set_ylabel('Waiting Time')
    ax1.legend(prop={"size": FONT_SIZE})

    # Plot queue length
    ax2.plot(avg_queue_lengths, alpha=1, label='Average queue length')
    ax2.fill_between(range(len(avg_queue_lengths)),
                     avg_queue_lengths - conf_interval_queue,
                     avg_queue_lengths + conf_interval_queue,
                     alpha=0.2)
    ax2.set_title('Average Queue Length Over Time')
    ax2.set_xlabel('Customer Index')
    ax2.set_ylabel('Queue Length')
    ax2.legend(prop={"size": FONT_SIZE})

    plt.tight_layout()
    plt.savefig(f'images/{config_name}_metrics_{queue_method.print_name}.pdf')
    plt.close()

def compare_queue_configurations(queue_method, results):
    """
    Create comprehensive and visually engaging comparison plots with all configurations.

    Args:
        results: Dictionary of simulation results
    """
    # Prepare data
    configs = list(results.keys())
    
    # Create a larger figure with more subplots for comprehensive comparison
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    service_time_dist = list(results.keys())[0].split('/')[1]

    # Box plot for waiting times
    waiting_times_data = [results[c]['waiting_times_runs'] for c in configs]
    axs[0, 0].boxplot(waiting_times_data, labels=configs)
    axs[0, 0].set_title(f'{queue_method.print_name}: Waiting Times Distribution')
    axs[0, 0].set_ylabel('Waiting Time')
    axs[0, 0].set_xticklabels(configs, rotation=45)

    # Error bar plot for mean waiting times with confidence intervals
    mean_waiting_times = [np.mean(results[c]['waiting_times_runs']) for c in configs]
    confidence_intervals = [results[c]['confidence_interval'] for c in configs]
    axs[0, 1].errorbar(configs, mean_waiting_times,
                       yerr=[(ci[1]-ci[0])/2 for ci in confidence_intervals],
                       fmt='o', capsize=5)
    axs[0, 1].set_title(f'{queue_method.print_name}: Mean Waiting Times')
    axs[0, 1].set_ylabel('Average Waiting Time')
    axs[0, 1].set_xticklabels(configs, rotation=45)

    # Average queue length
    avg_queue_lengths = [
        np.mean([np.mean(run) for run in results[c]['queue_length']]) for c in configs]
    axs[1, 0].bar(configs, avg_queue_lengths)
    axs[1, 0].set_title(f'{queue_method.print_name}: Average Queue Length')
    axs[1, 0].set_ylabel('Queue Length')
    axs[1, 0].set_xticklabels(configs, rotation=45)

    # Violin plot for server utilization
    axs[1, 1].violinplot([results[c]['utilization_runs'] for c in configs],
                         positions=range(len(configs)),
                         showmeans=True)
    axs[1, 1].set_title(f'{queue_method.print_name}: Utilization Distribution')
    axs[1, 1].set_xticks(range(len(configs)))
    axs[1, 1].set_xticklabels(configs, rotation=45)
    axs[1, 1].set_ylabel('Utilization')

    plt.tight_layout()
    plt.savefig(f'images/config_comparison_M{service_time_dist}X_{queue_method.print_name}.pdf')
    plt.close(fig)

def plot_consolidated_queue_metrics(queue_method, results, rho):
    """
    Create consolidated time series plots of waiting times and queue length for all configurations.

    Args:
        results: Dictionary of simulation results for different configurations
        rho: System load parameter for theoretical waiting time calculation
    """
    # Create a figure with two subplots for waiting times and queue length
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # Color palette for different configurations
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    x_axis_len = min([len(run) for run in results[list(results.keys())[-1]]['waiting_times_list']])

    # Plot for each configuration
    for (config, data), color in zip(results.items(), colors):
        # Extract server number from configuration name
        n_servers = int(config.split('/')[-1])
        
        # Consolidate waiting times
        waiting_times_list = data['waiting_times_list']
        min_length_waiting = min([len(run) for run in waiting_times_list])
        timings = [avg_waiting_times[:min_length_waiting] for avg_waiting_times in waiting_times_list]
        avg_waiting_times = np.mean(timings, axis=0)
        std_waiting_times = np.std(timings, axis=0)
        conf_interval_waiting = 1.96 * std_waiting_times / np.sqrt(len(waiting_times_list))
        
        # Waiting times plot
        ax1.plot(avg_waiting_times, color=color, label=config, linewidth=2, alpha=0.8)
        ax1.fill_between(range(len(avg_waiting_times)),
                         avg_waiting_times - conf_interval_waiting,
                         avg_waiting_times + conf_interval_waiting,
                         color=color, alpha=0.2)
        
        service_time_dist = list(results.keys())[0].split('/')[1]

        # Theoretical waiting time
        if queue_method != PriorityQueueSimulation and service_time_dist == 'M':
            ax1.plot(np.ones(x_axis_len) * rho/(n_servers * (1-rho)),
                    color=color, linestyle='--', alpha=0.5)
        
        # Queue length processing
        queue_length_list = data['queue_length']
        min_length_queue = min([len(run) for run in queue_length_list])
        queue_lengths = [avg_queue_lengths[:min_length_queue] for avg_queue_lengths in queue_length_list]
        avg_queue_lengths = np.mean(queue_lengths, axis=0)
        std_queue_lengths = np.std(queue_lengths, axis=0)
        conf_interval_queue = 1.96 * std_queue_lengths / np.sqrt(len(queue_length_list))
        
        # Queue length plot
        ax2.plot(avg_queue_lengths, color=color, label=config, linewidth=2, alpha=0.8)
        ax2.fill_between(range(len(avg_queue_lengths)),
                         avg_queue_lengths - conf_interval_queue,
                         avg_queue_lengths + conf_interval_queue,
                         color=color, alpha=0.2)
    
    # Finalize waiting times plot
    ax1.set_title(f'{queue_method.print_name}: Average Waiting Times')
    ax1.set_xlabel('Customer Index')
    ax1.set_ylabel('Waiting Time')
    ax1.legend(title='Configuration', prop={'size': FONT_SIZE * 1.2})
    
    # Finalize queue length plot
    ax2.set_title(f'{queue_method.print_name}: Average Queue Length')
    ax2.set_xlabel('Customer Index')
    ax2.set_ylabel('Queue Length')
    ax2.legend(title='Configuration', prop={'size': FONT_SIZE * 1.2})
    
    plt.tight_layout()
    plt.savefig(f'images/M{service_time_dist}X_{queue_method.print_name}_metrics.pdf')
    plt.close(fig)

def compare_server_numbers(queue_method, rho, service_rate, service_time_dist, n_runs, max_time=1e8):
    """
    Compare different server configurations for a given queue method.
    """

    configs = [
        (1, rho * service_rate, service_time_dist),  # M/M/1
        (2, 2 * rho * service_rate, service_time_dist),  # M/M/2
        (4, 4 * rho * service_rate, service_time_dist),  # M/M/4
    ]

    # Store results for all configurations
    results = {}
    for servers_number, arrival_rate, dist in configs:
        key = f"M/{dist}/{servers_number}"
        results[key] = run_simulation(
            queue_method, n_runs, servers_number, arrival_rate, service_rate, max_time, service_dist=dist)

    # Compute statistics for each configuration
    statistics = {}
    for config, data in results.items():
        mean_waiting_time = np.mean(data['waiting_times_runs'])
        confidence_interval = stats.t.interval(
            0.95,
            len(data['waiting_times_runs'])-1,
            loc=mean_waiting_time,
            scale=stats.sem(data['waiting_times_runs'])
        )

        statistics[config] = {
            'mean_waiting_time': mean_waiting_time,
            'mean_utilization': np.mean(data['utilization_runs']),
            'confidence_interval': confidence_interval,
            'waiting_times_runs': data['waiting_times_runs'],
            'utilization_runs': data['utilization_runs'],
            'queue_length': data['queue_length'],
            'waiting_times_list': data['waiting_times_list']
        }

    # Create comparison plots
    compare_queue_configurations(queue_method, statistics)
    
    # Create consolidated time series plots
    plot_consolidated_queue_metrics(queue_method, results, rho)

if __name__ == "__main__":
    
    params = {'legend.fontsize': 'x-large',
            #'figure.figsize': (15, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'x-large',
            'ytick.labelsize':'x-large',
            }
    
    font = {'size': FONT_SIZE}

    plt.rc('font', **font)
    plt.rcParams.update(params)

    rho = 0.9  # System load
    service_rate = 1.0  # Service rate
    n_runs = 1000
    max_time = 1e3

    # (num_servers, arrival_rate, distribution)
    compare_server_numbers(QueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time)
    compare_server_numbers(PriorityQueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time)

    compare_server_numbers(QueueSimulation, rho, service_rate, 'D', n_runs, max_time=max_time)
    compare_server_numbers(PriorityQueueSimulation, rho, service_rate, 'D', n_runs, max_time=max_time)

    compare_server_numbers(QueueSimulation, rho, service_rate, 'H', n_runs, max_time=max_time)
    compare_server_numbers(PriorityQueueSimulation, rho, service_rate, 'H', n_runs, max_time=max_time)