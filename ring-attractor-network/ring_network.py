import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 256  # Number of neurons
tau = 1.5  # Time constant
J0 = 0.5  # Baseline connectivity strength
J1 = 2.5  # Modulation amplitude of the cosine term
I0 = 2.0  # Baseline external input
I1 = 0.05  # Modulated external input amplitude
theta_0 = -1.2345# the angle selected by the external input
time_span = (0, 20)  # Start and end time for the simulation

# Neuron positions on the ring
theta = np.linspace(-np.pi, np.pi, N, endpoint=False)

# Define the external input I_i^0 as a function of theta (same dimension as theta)
I_e = I0 + I1 * np.cos(theta - theta_0)

# Define the weight matrix using the J function
def J(theta_diff):
    return J0 + J1 * np.cos(theta_diff)

# Compute the weight matrix w_ij
w = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        theta_diff = theta[i] - theta[j]
        w[i, j] = J(theta_diff) / N

# Rectified linear function
def F(x):
    return np.maximum(0, x)

# Define the system of ODEs
def ring_network_ode(t, u):
    # Compute the right-hand side of the equation
    du_dt = (-u + F(np.dot(w, u) + I_e)) / tau
    return du_dt

# Random initial activity levels for each neuron
perturbation = 0.05
u0 = perturbation * np.random.randn(N)

# Time points at which to evaluate the solution
time_points = np.linspace(0, 20, 1000)

# Solve the ODE system
solution = solve_ivp(ring_network_ode, time_span, u0, method='RK45', t_eval=time_points)
u_final = solution.y[:, -1]
peak = np.argmax(u_final) * (2 * np.pi / N) - np.pi

# Preparing data for the 3D plot
T, Theta = np.meshgrid(time_points, theta)
U = solution.y  # Activity data

# Plotting in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(T, Theta, U, cmap='viridis', edgecolor='none')

# Adding labels
ax.set_xlabel("Time")
ax.set_ylabel("Neuron Position on Ring (θ)")
ax.set_zlabel("Activity (u)")
ax.set_title(f"Peak at {peak}")

# Color bar for reference
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Activity (u)")
plt.show()