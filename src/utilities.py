import simpy
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from QueueSimulation import QueueSimulation, PriorityQueueSimulation


__all__ = ['run_simulation', 'compare_queue_configurations', 
           'plot_consolidated_queue_metrics', 'create_advanced_visualizations', 
           'compare_server_numbers', 'create_configuration_heatmap', 
           'perform_welch_t_test', 'analyze_conf_int', 'plot_minimum_runs']

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
                                service_rate, service_dist, max_time/num_servers)
        env.run(until=max_time)

        result = sim.get_statistics()
        results['waiting_times_runs'].append(result['avg_waiting_time'])
        results['service_times_runs'].append(result['avg_service_time'])
        results['utilization_runs'].append(result['utilization'])
        results['samples_runs'].append(result['num_samples'])
        results['waiting_times_list'].append(result['waiting_times'])
        results['queue_length'].append(result['queue_length'])

    return results

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
    axs[0, 0].set_title(f'Waiting Times Distribution')
    axs[0, 0].set_ylabel('Waiting Time')
    axs[0, 0].set_xticklabels(configs, rotation=45)

    # Box plot for number of customers with outliers
    num_customers_data = [results[c]['num_samples'] for c in configs]
    axs[0, 1].boxplot(num_customers_data, labels=configs, showfliers=True)
    axs[0, 1].set_title(f'Number of Customers')
    axs[0, 1].set_ylabel('Mean Number of Customers')
    axs[0, 1].set_xticklabels(configs, rotation=45)

    # Violin plot for service times distribution
    service_times_data = [results[c]['service_times_runs'] for c in configs]
    axs[1, 0].violinplot(service_times_data, positions=range(len(configs)), showmeans=True)
    axs[1, 0].set_title(f'Service Times Distribution')
    axs[1, 0].set_xticks(range(len(configs)))
    axs[1, 0].set_xticklabels(configs, rotation=45)
    axs[1, 0].set_ylabel('Service Time')


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
    plt.figure(figsize=(12, 8))
    
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
        
        service_time_dist = list(results.keys())[0].split('/')[1]

        # Theoretical waiting time
        if queue_method != PriorityQueueSimulation and service_time_dist == 'M':
            plt.plot(np.ones(x_axis_len) * rho/(n_servers * (1-rho)),
                    color=color, linestyle='--', alpha=0.8, linewidth=2)
            
        if queue_method != PriorityQueueSimulation:
            std_waiting_times = np.std(timings, axis=0)
            conf_interval_waiting = 1.96 * std_waiting_times / np.sqrt(len(waiting_times_list))
            
            # Waiting times plot
            plt.plot(avg_waiting_times, color=color, label=config, linewidth=2, alpha=0.8)
            plt.fill_between(
                range(len(avg_waiting_times)), 
                avg_waiting_times - conf_interval_waiting,
                avg_waiting_times + conf_interval_waiting,
                color=color, alpha=0.2)
            
        elif queue_method == PriorityQueueSimulation and service_time_dist != 'H':
            plt.plot(avg_waiting_times, color=color, label=config, linewidth=1, alpha=0.2)
            
            window_size = 2 * int(np.log2(len(avg_waiting_times)))
            rolling_mean = np.convolve(avg_waiting_times, np.ones(window_size)/window_size, mode='valid')
            plt.plot(np.arange(window_size - 1, len(avg_waiting_times)), rolling_mean, color=color, linewidth=2)
        
        elif queue_method == PriorityQueueSimulation and service_time_dist == 'H':
            plt.plot(avg_waiting_times, color=color, label=config, linewidth=1, alpha=0.8)     

    
    # Finalize waiting times plot
    plt.title(f'Average Waiting Times')
    plt.xlabel('Customer Index')
    plt.ylabel('Waiting Time')
    plt.legend(title='Configuration')
    
    plt.tight_layout()
    plt.savefig(f'images/M{service_time_dist}X_{queue_method.print_name}_{rho}_metrics.pdf')
    plt.close()

def compare_server_numbers(queue_method, rho, service_rate, service_time_dist, n_runs, max_time=1e3):
    """
    Compare different server configurations for a given queue method.
    """
    if service_time_dist != 'H':
        configs = [
            (1, rho * service_rate, service_time_dist),  # M/M/1
            (2, 2 * rho * service_rate, service_time_dist),  # M/M/2
            (4, 4 * rho * service_rate, service_time_dist),  # M/M/4
        ]
    else:
        configs = [
            (1, (rho * 0.2)/(0.8), service_time_dist),  # M/M/1
            (2, (2 * rho * 0.2)/(0.8), service_time_dist),  # M/M/2
            (4, (4 * rho * 0.2)/(0.8), service_time_dist),  # M/M/4
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

        if queue_method == PriorityQueueSimulation:
            config += 'P'
        
        statistics[config] = {
            'mean_waiting_time': mean_waiting_time, #Averages across sims
            'waiting_times_runs': data['waiting_times_runs'], #List of averages
            'waiting_times_list': data['waiting_times_list'], #List of list of timings of each run
            'service_times_runs': data['service_times_runs'],
            'utilization_runs': data['utilization_runs'],
            'num_samples': data['samples_runs'],
            'queue_length': data['queue_length']
        }
   
    # Create consolidated time series plots
    plot_consolidated_queue_metrics(queue_method, results, rho)

    return statistics

def create_configuration_heatmap(rho_results):
    """
    Create a heatmap to compare waiting times across different configurations and system loads.
    
    Args:
        rho_results (dict): Simulation results for different system loads
    """
    # Prepare data for heatmap
    configurations = list(rho_results[list(rho_results.keys())[0]].keys())
    rho_values = list(rho_results.keys())
    
    # Create matrix of mean waiting times
    waiting_times_matrix = np.zeros((len(rho_values), len(configurations)))
    
    for i, rho in enumerate(rho_values):
        for j, config in enumerate(configurations):
            waiting_times = rho_results[rho][config]['waiting_times_runs']
            waiting_times_matrix[i, j] = np.mean(waiting_times)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(waiting_times_matrix.T, 
                annot=True, 
                cmap='YlGnBu', 
                yticklabels=configurations, 
                xticklabels=[f'ρ = {rho}' for rho in rho_values],
                fmt='.2f',
                cbar_kws={'label': 'Mean Waiting Time'})
    
    plt.title('Mean Waiting Times Across Configurations')
    plt.ylabel('Queue Configuration')
    plt.xlabel('System Load (ρ)')
    plt.tight_layout()
    plt.savefig('images/waiting_times_heatmap.pdf')
    plt.close()

def create_advanced_visualizations(comparisons):

    # Prepare DataFrame for visualization
    plot_data = []
    for (config, rho), stats in comparisons.items():
        plot_data.append({
            'Config 1': config, 
            'ρ' : rho,
            't-statistic': stats['t_statistic'],
            'p-value': stats['p_value']
        })
    
    df = pd.DataFrame(plot_data)
    
   # 1. Heatmap of p-values
    plt.figure(figsize=(10, 8))
    p_value_matrix = pd.pivot_table(df, values='p-value', index='Config 1', columns='ρ')
    sns.heatmap(p_value_matrix, annot=True, cmap='YlGnBu', 
                vmin=0, vmax=0.1, center=0.05)
    plt.title('Statistical Significance Heatmap')
    plt.tight_layout()
    plt.savefig('images/p_value_heatmap.pdf')
    plt.close()
    
def perform_welch_t_test(service_time_dist, n_runs, results, rho):
    baseline_config = f'M/{service_time_dist}/1'

    test_res = {}
    for config, data in results.items():
        if config != baseline_config and config.split('/')[1] == service_time_dist:
            
            waiting_times1 = np.average(data['waiting_times_runs'])
            waiting_times2 = np.average(results[baseline_config]['waiting_times_runs'])
            std1 = np.std(data['waiting_times_runs'])
            std2 = np.std(results[baseline_config]['waiting_times_runs'])

            t_stat = (waiting_times1 - waiting_times2) /  np.sqrt( (std1**2)/n_runs + (std2**2)/n_runs)
            dof = (std1**2/n_runs + std2**2/n_runs) / ((std1**2/(n_runs))**2/(n_runs-1) + (std2**2/(n_runs))**2/(n_runs-1))
            p_value = stats.t.sf(np.abs(t_stat), dof) * 2

            test_res[(config, rho)] = {
                't_statistic': t_stat,
                'p_value': p_value
            }

    return test_res

def analyze_conf_int(service_time_dist, rho_results, rho):

    rho_results = {k: v for k, v in rho_results.items() if k.split('/')[1]==service_time_dist and not k.endswith('P')}
    results = {}

    for (config, data) in rho_results.items():
        # Extract server number from configuration name
        results[config] = []

        # Consolidate waiting times
        waiting_times_list = data['waiting_times_list']
        min_length_waiting = min([len(run) for run in waiting_times_list])
        timings = [avg_waiting_times[:min_length_waiting] for avg_waiting_times in waiting_times_list]
        avg_waiting_times = np.mean(timings, axis=0)    

        service_time_dist = list(results.keys())[0].split('/')[1]

        std_waiting_times = np.std(timings, axis=0)

        for n in range(1, 1000):
            conf_interval_waiting = 1.96 * std_waiting_times / np.sqrt(n)
            lb = np.average(avg_waiting_times - conf_interval_waiting)
            ub = np.average(avg_waiting_times + conf_interval_waiting)
            results[config].append(np.array([lb, ub]))

    baseline = f'M/{service_time_dist}/1'
    baseline_lb = np.array(results[baseline])[:, 0]
    indexes = {}
    for config, data in results.items():
        if config == baseline:
            continue
        
        config_ub = np.array(results[config])[:, 1]
        index = np.argmax(baseline_lb > config_ub)
        indexes[(config, rho)] = index

    return indexes

def plot_minimum_runs(indexes, rho_range, service_time_dist, n_servers=[2, 4]):
    for n in n_servers:
        runs = [x for k, x in indexes.items() if k[0].split('/')[1] == service_time_dist and k[0].split('/')[2] == str(n)]
        plt.plot(rho_range, runs, '-o', label=f'M/{service_time_dist}/{n}')
        
    plt.xlabel(r"$\rho$")
    plt.ylabel("$n$")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"images/runs_required_{service_time_dist}.pdf")
    plt.close()
