import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

def perceptron_with_gaussian_init(X, y0, max_epochs=1000, eta= 0.1):
    N, P = X.shape  # N: number of features, P: number of samples
    w = np.random.normal(0, 1, N)  # Initialize weight vector using unit Gaussian (mean=0, std=1)
    converged = False  
    
    for epoch in range(max_epochs):
        error_count = 0
        
        for mu in range(P):
            if np.dot(w, X[:, mu]) * y0[mu] < 0: 
                w = w + eta * y0[mu] * X[:, mu] 
                error_count += 1
        
        # If no misclassified samples, the algorithm has converged
        if error_count == 0:
            converged = True
            return w, converged, epoch + 1
    
    # If maximum number of epochs is reached without convergence
    return w, converged, max_epochs

def collect_stats(N_values, P_values, runs=50):
    print(f"Collecting stats for N={N_values} and P={P_values} with {runs} runs for each (N, P) pair")
    stats = {}
    
    for N in N_values:
        for P in P_values:
            converged_count = 0
            total_epochs = 0
            
            for _ in range(runs):
                X = np.random.choice([-1, 1], size=(N, P))
                y0 = np.random.choice([-1, 1], size=P)
                
                w, converged, epochs = perceptron_with_gaussian_init(X, y0)
                
                if converged:
                    converged_count += 1
                total_epochs += epochs
            
            stats[(N, P)] = {
                "converged_percentage": (converged_count / runs) * 100,
                "average_epochs": total_epochs / runs
            }
    
    return stats

def compare_convergence_speed(N_values, P, runs=50):
    print(f"Comparing convergence speed for P={P} and N={N_values} with {runs} runs for each N")
    stats = {}
    
    for N in N_values:
        converged_count = 0
        total_epochs = 0
        
        for _ in range(runs):
            X = np.random.choice([-1, 1], size=(N, P))
            y0 = np.random.choice([-1, 1], size=P)
            
            _, converged, epochs = perceptron_with_gaussian_init(X, y0)
            
            if converged:
                converged_count += 1
                total_epochs += epochs
        
        stats[N] = {
            "converged_percentage": (converged_count / runs) * 100,
            "average_epochs": total_epochs / runs
        }
    
    return stats

def plot_convergence_probability(N, alpha_range, runs=50):
    print(f"Calculating convergence probability for N={N} and alpha range {alpha_range}")
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], 50)
    convergence_probs = []

    for alpha in alpha_values:
        P = int(alpha * N)
        converged_count = 0

        for _ in range(runs):
            X = np.random.choice([-1, 1], size=(N, P))
            y0 = np.random.choice([-1, 1], size=P)
            
            _, converged, _ = perceptron_with_gaussian_init(X, y0)
            
            if converged:
                converged_count += 1
        
        convergence_prob = converged_count / runs
        convergence_probs.append(convergence_prob)

    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, convergence_probs, '-o')
    plt.xlabel('α (P/N)')
    plt.ylabel('Convergence Probability')
    plt.title(f'Convergence Probability vs α (N={N})')
    plt.savefig(f'perceptron-results/convergence-probability-N={N}.png')
    plt.grid(True)

    return alpha_values, convergence_probs

def calculate_margin(w, X, y0):
    margins = []
    for mu in range(X.shape[1]):  # Iterate over columns (samples)
        margin = abs(np.dot(w, X[:, mu]) * y0[mu])
        margins.append(margin)
    return min(margins)  # Return the minimum margin


def calculate_mean_margin(N, P, runs=50):
    margins = []
    for _ in range(runs):
        X = np.random.choice([-1, 1], size=(N, P))
        y0 = np.random.choice([-1, 1], size=P)
        w, converged, _ = perceptron_with_gaussian_init(X, y0)
        if converged and P > 0:
            margins.append(calculate_margin(w, X, y0))
    return np.mean(margins) if margins else 0

def plot_margin_histogram(N, alpha_range, runs=50):
    print(f"Calculating mean margins for N={N} and alpha range {alpha_range}")
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], 50)
    mean_margins = []

    for alpha in alpha_values:
        P = int(alpha * N)
        mean_margin = calculate_mean_margin(N, P, runs)
        mean_margins.append(mean_margin)

    plt.figure(figsize=(10, 6))
    plt.bar(alpha_values, mean_margins, width=0.1)
    plt.xlabel('α (P/N)')
    plt.ylabel('Mean Margin')
    plt.title(f'Mean Margin Histogram H(P,N) for N={N}')
    plt.grid(True)
    plt.savefig(f'perceptron-results/margin-histogram-N={N}.png')
    plt.close()

    return alpha_values, mean_margins

