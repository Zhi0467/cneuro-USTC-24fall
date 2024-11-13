import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 256  # Number of neurons
tau = 1.0  # Time constant
J0 = 0.0  # Baseline connectivity strength
I0 = 1.0  # Baseline external input
I1 = 0.0  # Modulated external input amplitude
time_span = (0, 20)  # Start and end time for the simulation

# Neuron positions on the ring
theta = np.linspace(-np.pi, np.pi, N, endpoint=False)
theta_0 = theta[0]  # Reference angle

# Define the external input I_i^0 as a function of theta (same dimension as theta)
I0_i = I0 + I1 * np.cos(theta - theta_0)

# Rectified linear function
def F(x):
    return np.maximum(0, x)

# Function to compute weight matrix given J1
def compute_weights(J1):
    def J(theta_diff):
        return J0 + J1 * np.cos(theta_diff)
    
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            theta_diff = theta[i] - theta[j]
            w[i, j] = J(theta_diff) / N
    return w

# Define the system of ODEs
def ring_network_ode(t, u, w):
    du_dt = (-u + F(np.dot(w, u) + I0_i)) / tau
    return du_dt

# Simulation function
def simulate(J1, perturbation=0.1):
    # Compute weight matrix with given J1
    w = compute_weights(J1)

    # Initial activity levels with small random perturbations around 0
    u0 = np.ones(N)
    u0 += perturbation * np.random.randn(N)
    
    # Time points at which to evaluate the solution
    time_points = np.linspace(0, 20, 1000)

    # Solve the ODE system
    solution = solve_ivp(ring_network_ode, time_span, u0, args=(w,), method='RK45', t_eval=time_points)

    # Prepare data for plotting
    T, Theta = np.meshgrid(time_points, theta)
    U = solution.y  # Activity data

    return T, Theta, U

# Plot function for 3D surface
def plot_3d(T, Theta, U, title):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T, Theta, U, cmap='viridis', edgecolor='none')
    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron Position (θ)")
    ax.set_zlabel("Activity (u)")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Activity (u)")
    plt.show()

# Run simulations
T1, Theta1, U1 = simulate(J1=1.5)  # Case J1 <= 2 (expected to be stable)
T2, Theta2, U2 = simulate(J1=2.5)  # Case J1 > 2 (expected to be unstable)
T3, Theta3, U3 = simulate(J1=5.0)  # Case J1 > 2 (expected to be unstable)

# Plot results
plot_3d(T1, Theta1, U1, "Stable Homogeneous Solution (J1 = 1.5)")
plot_3d(T2, Theta2, U2, "Unstable Homogeneous Solution (J1 = 2.5)")
plot_3d(T2, Theta2, U2, "Unstable Homogeneous Solution (J1 = 5.0)")