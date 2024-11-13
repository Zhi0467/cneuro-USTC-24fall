import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
"""
we sample some (J_0, J_1) data points and use help functions to classify them 
to determine the final state of the network
assumptions:
- random small initial population activity << tiny noisy 
- I_0 = 1.0, I_1 = 0.05, theta_0 = 0.0 << small modulation
"""

def is_localized_bump(activity, threshold_ratio=1.5, spread_threshold=0.7):
    """
    Determine if an activity profile is a localized bump.
    """
    # Calculate basic properties of the activity profile
    max_activity = np.max(activity)
    mean_activity = np.mean(activity)
    
    # Check if the peak is significantly higher than the mean
    if max_activity < threshold_ratio * mean_activity:
        return False  # Peak is not distinct enough
    
    # Find the indices around the peak that have high activity
    peak_index = np.argmax(activity)
    high_activity_indices = np.where(activity > mean_activity)[0]
    
    # Calculate the spread of high activity around the peak
    spread = np.sum(np.abs(high_activity_indices - peak_index) < len(activity) * spread_threshold) / len(activity)
    
    # If the spread is small, it indicates localization
    return spread < spread_threshold

def is_unimodal(arr):
    # Find the index of the maximum value
    max_idx = np.argmax(arr)
    
    # Check if it strictly increases before the max and strictly decreases after
    if np.all(np.diff(arr[:max_idx]) > 0) and np.all(np.diff(arr[max_idx:]) < 0):
        return True
    return False

def has_significant_negative_section(activity, negative_threshold=0.05):
    """
    Check if an activity profile has a significant negative section.
    """
    negative_fraction = np.sum(activity < 0) / len(activity)
    return negative_fraction > negative_threshold

def too_big(activity, threshold = 10.0):
    """
    check if an activity profile is too big
    """
    std_activity = np.std(activity)
    return std_activity >= threshold

def max_too_big(activity, threshold):
    max_activity = np.max(activity)
    return max_activity >= threshold

def is_homogenous(activity, threshold = 1e-1):

    std_dev = np.std(activity)
    return std_dev <= threshold

# Define simulation parameters
tau = 1.0
theta = np.linspace(-np.pi, np.pi, 256)
I0 = 1.0
I1 = 0.05
theta_0 = 0.0
I_e = I0 + I1 * np.cos(theta - theta_0)
N = len(theta)

# Set up ranges for J0 and J1
J0_values = np.linspace(-1, 4, 50)  
J1_values = np.linspace(0, 10, 50)

# Initialize the phase diagram matrix to store the results
phase_diagram = np.zeros((len(J0_values), len(J1_values)))

def ring_network_rhs(t, u, J0, J1):
    """RHS of the ring network ODE system."""
    # Compute weights with connectivity pattern J
    u_activity = np.maximum(u, 0)  # Apply rectified linear function
    J = J0 + J1 * np.cos(theta[:, None] - theta[None, :])
    W = (1 / N) * J
    du_dt = -u + np.dot(W, u_activity) + I_e
    return du_dt / tau

# Iterate over J0 and J1 values to simulate the system and classify results
for i, J0 in enumerate(J0_values):
    for j, J1 in enumerate(J1_values):
        # Set initial condition (e.g., small random noise around zero)
        u0 = 0.01 * np.random.randn(N)

        # Run the simulation
        sol = solve_ivp(ring_network_rhs, [0, 20], u0, args=(J0, J1), t_eval=np.linspace(0, 20, 1000))

        # Analyze the final state
        u_final = sol.y[:, -1]

        # Classify the final state

        if has_significant_negative_section(u_final) or too_big(u_final):
            phase_diagram[i, j] = 2
        else:
            phase_diagram[i, j] = 1  
        # some of the bumps with large mean are misclassified 
        # so we correct it below

        if is_localized_bump(u_final) or is_unimodal(u_final):
            phase_diagram[i, j] = 1

        if is_homogenous(u_final):
            phase_diagram[i, j] = 0
        
        if max_too_big(u_final, threshold= 150.0):
            phase_diagram[i, j] = 2
        
        

# Generate scatter plot for phase diagram
colors = ['blue', 'orange', 'red']  # Color codes for each state
plt.figure(figsize=(8, 6))

for i, J0 in enumerate(J0_values):
    for j, J1 in enumerate(J1_values):
        # Select the color based on the classification result
        color = colors[int(phase_diagram[i, j])]
        plt.scatter(J0, J1, color=color, s=30)

# Customize the plot
plt.xlabel("J0")
plt.ylabel("J1")
plt.title("Phase Diagram of Ring Network - Scatter Plot")
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Homogenous'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Bump'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Unstable or unclassified')
], loc="upper right")
plt.grid(True)
plt.show()
