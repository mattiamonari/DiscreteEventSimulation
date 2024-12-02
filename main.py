import simpy
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    axs[0, 0].set_title(f'{queue_method.print_name}: Waiting Times Distribution')
    axs[0, 0].set_ylabel('Waiting Time')
    axs[0, 0].set_xticklabels(configs, rotation=45)

    # Box plot for number of customers with outliers
    num_customers_data = [results[c]['num_samples'] for c in configs]
    axs[0, 1].boxplot(num_customers_data, labels=configs, showfliers=True)
    axs[0, 1].set_title(f'{queue_method.print_name}: Number of Customers')
    axs[0, 1].set_ylabel('Mean Number of Customers')
    axs[0, 1].set_xticklabels(configs, rotation=45)

    # Violin plot for service times distribution
    service_times_data = [results[c]['service_times_runs'] for c in configs]
    axs[1, 0].violinplot(service_times_data, positions=range(len(configs)), showmeans=True)
    axs[1, 0].set_title(f'{queue_method.print_name}: Service Times Distribution')
    axs[1, 0].set_xticks(range(len(configs)))
    axs[1, 0].set_xticklabels(configs, rotation=45)
    axs[1, 0].set_ylabel('Service Time')

    # Queue Length Variance Plot
    # print(np.var(results[0]['queue_length']))
    # queue_length_variance = [np.var(results[c]['queue_length']) for c in configs]
    # axs[1, 1].bar(configs, queue_length_variance)
    # axs[1, 1].set_title(f'{queue_method.print_name}: Queue Length Variance')
    # axs[1, 1].set_ylabel('Variance of Queue Length')
    # axs[1, 1].set_xticklabels(configs, rotation=45)

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
        
        service_time_dist = list(results.keys())[0].split('/')[1]

        # Theoretical waiting time
        if queue_method != PriorityQueueSimulation and service_time_dist == 'M':
            ax1.plot(np.ones(x_axis_len) * rho/(n_servers * (1-rho)),
                    color=color, linestyle='--', alpha=0.8, label=config + ' (theoretical)', linewidth=2)
            
        if queue_method != PriorityQueueSimulation:
            std_waiting_times = np.std(timings, axis=0)
            conf_interval_waiting = 1.96 * std_waiting_times / np.sqrt(len(waiting_times_list))
            
            # Waiting times plot
            ax1.plot(avg_waiting_times, color=color, label=config, linewidth=2, alpha=0.8)
            ax1.fill_between(
                range(len(avg_waiting_times)), 
                avg_waiting_times - conf_interval_waiting,
                avg_waiting_times + conf_interval_waiting,
                color=color, alpha=0.2)
            
        elif queue_method == PriorityQueueSimulation and service_time_dist != 'H':
            ax1.plot(avg_waiting_times, color=color, label=config, linewidth=1, alpha=0.2)
            
            window_size = 2 * int(np.log2(len(avg_waiting_times)))
            rolling_mean = np.convolve(avg_waiting_times, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(np.arange(window_size - 1, len(avg_waiting_times)), rolling_mean, color=color, label=config + ' (smoothed)', linewidth=2)
        
        elif queue_method == PriorityQueueSimulation and service_time_dist == 'H':
            ax1.plot(avg_waiting_times, color=color, label=config, linewidth=1, alpha=0.8)     
        
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
    plt.savefig(f'images/M{service_time_dist}X_{queue_method.print_name}_{rho}_metrics.pdf')
    plt.close(fig)

def compare_server_numbers(queue_method, rho, service_rate, service_time_dist, n_runs, max_time=1e3):
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

        # statistics[config] = {
        #     'mean_waiting_time': mean_waiting_time,
        #     'variance_waiting_time':  np.var(data['waiting_times_runs']), #not used
        #     'mean_service_time' : np.mean(data['service_times_runs']),
        #     'variance_service_time': np.var(data['service_times_runs']), #not used
        #     'confidence_interval': confidence_interval,
        #     'mean_num_samples': np.average(data['samples_runs'])    , #not used
        #     'waiting_times_runs': data['waiting_times_runs'],
        #     'utilization_runs': data['utilization_runs'],
        #     'queue_length': data['queue_length'],
        #     'waiting_times_list': data['waiting_times_list']
        # }

        # statistics[config] = {
        #     'mean_waiting_time': mean_waiting_time,
        #     'mean_utilization': np.mean(data['utilization_runs']),
        #     'confidence_interval': confidence_interval,
        #     'waiting_times_runs': data['waiting_times_runs'],
        #     'utilization_runs': data['utilization_runs'],
        #     'queue_length': data['queue_length'],
        #     'waiting_times_list': data['waiting_times_list']
        # }

        if queue_method == PriorityQueueSimulation:
            config += 'P'
        
        statistics[config] = {
            'mean_waiting_time': mean_waiting_time,
            'mean_utilization': np.mean(data['utilization_runs']),
            'service_times_runs': data['service_times_runs'],
            'confidence_interval': confidence_interval,
            'waiting_times_runs': data['waiting_times_runs'],
            'utilization_runs': data['utilization_runs'],
            'num_samples': data['samples_runs'],
            'queue_length': data['queue_length']
        }

    # Create comparison plots
    #compare_queue_configurations(queue_method, statistics)
    
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
    plt.figure(figsize=(12, 8))
    sns.heatmap(waiting_times_matrix, 
                annot=True, 
                cmap='YlGnBu', 
                xticklabels=configurations, 
                yticklabels=[f'ρ = {rho}' for rho in rho_values],
                fmt='.2f',
                cbar_kws={'label': 'Mean Waiting Time'})
    
    plt.title('Mean Waiting Times Across Configurations and System Loads')
    plt.xlabel('Queue Configuration')
    plt.ylabel('System Load (ρ)')
    plt.tight_layout()
    plt.savefig('images/waiting_times_heatmap.pdf')
    plt.close()

def create_advanced_visualizations(comparisons):

    # Prepare DataFrame for visualization
    plot_data = []
    for (rho1, rho2, config), stats in comparisons.items():
        plot_data.append({
            'Config 1': config, 
            'ρ' : rho1,
            't-statistic': stats['t_statistic'],
            'p-value': stats['p_value'],
            'Required Sample Size': stats['samples_size']
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
    
    # 2. Scatter plot of t-statistic vs Required Sample Size
    plt.figure(figsize=(10, 6))
    plt.scatter(df['t-statistic'], df['Required Sample Size'], 
                c=df['p-value'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='p-value')
    plt.xlabel('t-statistic')
    plt.ylabel('Required Sample Size')
    plt.title('t-statistic vs Required Sample Size')
    plt.tight_layout()
    plt.savefig('images/t_statistic_sample_size.pdf')
    plt.close()

    # 3. Violin plot of sample size requirements
    plt.figure(figsize=(10, 8))
    sample_size_matrix = pd.pivot_table(df, values='Required Sample Size', index='Config 1', columns='ρ')
    sns.heatmap(sample_size_matrix, annot=True, cmap='YlOrRd', fmt='g')
    plt.title('Required Sample Size Between Configurations')
    plt.tight_layout()
    plt.savefig('images/sample_size_heatmap.pdf')
    plt.close()

def calculate_required_sample_size(
    effect_size=0.5, 
    significance_level=0.05, 
    power=0.8
):
    """
    Calculate required sample size for statistical power
    
    Parameters:
    - effect_size: Minimum detectable difference
    - significance_level: Type I error rate
    - power: 1 - Type II error rate
    
    Returns:
    - Minimum sample size
    """
    # Z-scores for significance and power
    z_alpha = stats.norm.ppf(1 - significance_level/2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size calculation
    sample_size = int(np.ceil(
        ((z_alpha + z_beta) / effect_size)**2
    ))
    
    return sample_size


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

    service_rate = 1  # Service rate
    n_runs = 50
    max_time = 3e3
    baseline_rho = 0.8
    rho_range = [baseline_rho, 0.85, 0.9, 0.95]
    

    results = {}
    rho_results = {}
    
    for rho in rho_range:
        rho_results[rho] = compare_server_numbers(QueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time)
        rho_results[rho].update(compare_server_numbers(PriorityQueueSimulation, rho, service_rate, 'M', n_runs, max_time=max_time))
        rho_results[rho].update(compare_server_numbers(QueueSimulation, rho, service_rate, 'D', n_runs, max_time=max_time))
        rho_results[rho].update(compare_server_numbers(QueueSimulation, rho, service_rate, 'H', n_runs, max_time=max_time))

    comparisons = {}
    for config in rho_results[baseline_rho].keys():
        for i1, rho1 in enumerate(rho_range):
                if rho1 == baseline_rho:
                    continue

                t_stat, p_value = stats.ttest_ind(
                    rho_results[rho1][config]['waiting_times_runs'], 
                    rho_results[baseline_rho][config]['waiting_times_runs']
                )

                mean1 = np.mean(rho_results[rho1][config]['waiting_times_runs'])
                mean2 = np.mean(rho_results[baseline_rho][config]['waiting_times_runs'])
            
                # Pooled standard deviation
                std1 = np.std(rho_results[rho1][config]['waiting_times_runs'])
                std2 = np.std(rho_results[baseline_rho][config]['waiting_times_runs'])

                pooled_std = np.sqrt(
                    ((n_runs-1) * std1**2 + (n_runs-1) * std2**2) / (2 * n_runs - 2)
                )
                
                # Cohen's d effect size
                effect_size = abs(mean1 - mean2) / pooled_std

                # Calculate required sample size
                n = calculate_required_sample_size(
                    effect_size=effect_size,
                    power=0.9,
                    significance_level=0.01
                )

                comparisons[(rho1, baseline_rho, config)] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'samples_size': n
                }
    
    create_configuration_heatmap(rho_results)
    create_advanced_visualizations(comparisons)

    # compare_queue_configurations(QueueSimulation, results)
