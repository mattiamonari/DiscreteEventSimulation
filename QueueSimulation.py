import simpy
import random
import numpy as np
import scipy.stats as stats

class QueueSimulation:
    def __init__(self, env, num_servers, arrival_rate, service_rate, service_dist='M', max_time=100000):
        self.env = env
        self.num_servers = num_servers
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.service_dist = service_dist
        self.max_time = max_time
        
        # Create server resource
        self.server = simpy.Resource(env, capacity=num_servers)
        
        # Statistics collection
        self.waiting_times = []
        self.service_times = []
        self.arrivals = 0
        self.completed = 0
        
        # Start the process
        self.env.process(self.customer_arrival())
        
    def generate_service_time(self):
        if self.service_dist == 'M':  # Exponential
            return random.expovariate(self.service_rate)
        elif self.service_dist == 'D':  # Deterministic
            return 1.0 / self.service_rate
        elif self.service_dist == 'H':  # Hyperexponential
            if random.random() < 0.75:
                return random.expovariate(1.0)  # Mean = 1.0
            else:
                return random.expovariate(0.2)  # Mean = 5.0
    
    def customer_arrival(self):
        """Generate customer arrivals."""
        while self.env.now < self.max_time:
            # Generate next arrival
            interarrival_time = random.expovariate(self.arrival_rate)
            yield self.env.timeout(interarrival_time)
            
            # Start customer service process
            self.arrivals += 1
            self.env.process(self.customer_service())
    
    def customer_service(self):
        """Customer service process."""
        arrival_time = self.env.now
        service_time = self.generate_service_time()
        
        # Request server
        with self.server.request() as request:
            yield request
            
            # Calculate waiting time
            waiting_time = self.env.now - arrival_time
            self.waiting_times.append(waiting_time)
            self.service_times.append(service_time)
            
            # Service the customer
            yield self.env.timeout(service_time)
            self.completed += 1
    
    #TODO UNDERSTANDE REGARDING THE CONFIDENCE INTERVAL
    def get_statistics(self):
        """Calculate statistics from the simulation."""
        
        return {
            'waiting_times': self.waiting_times,
            'avg_waiting_time': np.mean(self.waiting_times),
            'avg_service_time': np.mean(self.service_times),
            'num_samples': len(self.waiting_times),
            'utilization': np.mean(self.service_times) * self.arrival_rate / self.num_servers
        }
