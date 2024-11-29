import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import simpy
import scipy.stats as stats
import numpy as np
from QueueSimulation import QueueSimulation, PriorityQueueSimulation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random 

def run_simulation(queue_method, num_runs, num_servers, arrival_rate, service_rate, service_dist='M', max_time=100000):
    """Run multiple simulation replications."""
    results = {
        'waiting_times_runs': [],
        'service_times_runs': [],
        'utilization_runs': [],
        'samples_runs' : [],
        'waiting_times_list': []
    }

    for _ in range(num_runs):
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
    
    return results

def compare_queue_configurations(queue_method,results):
    """
    Create comprehensive comparison plots for queue configurations.
    
    Args:
        results: Dictionary of simulation results
    """
    # Prepare data
    configs = list(results.keys())
    mean_waiting_times = [results[c]['mean_waiting_time'] for c in configs]
    waiting_time_variances = [results[c]['variance_waiting_time'] for c in configs]
    mean_utilizations = [results[c]['mean_utilization'] for c in configs]
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Mean Waiting Times
    axs[0].bar(configs, mean_waiting_times)
    axs[0].set_title('Mean Waiting Times')
    axs[0].set_ylabel('Average Waiting Time')
    
    # Waiting Time Variances
    axs[1].bar(configs, waiting_time_variances)
    axs[1].set_title('Waiting Time Variances')
    axs[1].set_ylabel('Variance')
    
    # Mean Utilizations
    axs[2].bar(configs, mean_utilizations)
    axs[2].set_title('Mean Utilizations')
    axs[2].set_ylabel('Utilization')
    axs[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'images/queue_configuration_comparison_{queue_method.__name__}.pdf')
    
    return fig

def plot_time_series(queue_method, waiting_times, window_size=25):
    """
    Create a time series plot of waiting times with rolling average.
    
    Args:
        simulation: QueueSimulation object
        window_size: Size of the rolling average window
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate rolling average
    rolling_mean = np.convolve(waiting_times, 
                              np.ones(window_size)/window_size, 
                              mode='valid')
    
    # Plot raw data and rolling average
    plt.plot(waiting_times, alpha=0.3, label='Raw waiting times')
    plt.plot(np.arange(window_size-1, len(waiting_times)), 
             rolling_mean, 
             label=f'Rolling average (window={window_size})')
    
    plt.title('Waiting Times Over Time')
    plt.xlabel('Customer Index')
    plt.ylabel('Waiting Time')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'images/MM1_time_series_{queue_method.__name__}.pdf')


def compare_server_numbers(queue_method):
    rho = 0.9  # System load
    server_capacity = 1.0 # In the assignment this is called server capacity. Why?
    n_runs = 250
    
    #(num_servers, arrival_rate, distribution)
    configs = [
        (1, rho * server_capacity, 'M'),  # M/M/1
        (2, 2 * rho * server_capacity, 'M'),  # M/M/2
        (4, 4 * rho * server_capacity, 'M'),  # M/M/4
    ]

    results = {}
    for servers_number, lambda_, dist in configs:
        key = f"M/{dist}/{servers_number}"
        results[key] = run_simulation(queue_method,n_runs, servers_number, lambda_, server_capacity, dist, n_runs)


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
            'variance_service_time': mean_service_time,
            'variance_utilization': variance_utilization,
            'confidence_interval': confidence_interval,
            'mean_num_samples': num_samples
        }

    compare_queue_configurations(queue_method,statistics)

    random_index = int(random.random() * statistics['M/M/1']['mean_num_samples'])

    print(results['M/M/1']['waiting_times_runs'][random_index])
    
    plot_time_series(queue_method, results['M/M/1']['waiting_times_list'][random_index], window_size=5)


if __name__ == "__main__":
    compare_server_numbers(QueueSimulation)
    compare_server_numbers(PriorityQueueSimulation)
