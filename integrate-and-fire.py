import numpy as np
import matplotlib.pyplot as plt

# Parameters
tau = 20  # membrane time constant in ms
tau_r = 2  # refractory period in ms
V_threshold = 1  # threshold voltage for spike
I_e_values = np.linspace(1, 20, 100)  # range of external currents

# Firing rate without refractory period
def firing_rate_no_refractory(I_e, tau, V_threshold):
    # Time to reach threshold: V(t) = V_threshold, solve for t
    t_spike = tau * np.log(I_e / (I_e - V_threshold))
    # Firing rate is inverse of time per spike
    return 1 / t_spike

# Firing rate with refractory period
def firing_rate_with_refractory(I_e, tau, V_threshold, tau_r):
    t_spike = tau * np.log(I_e / (I_e - V_threshold))
    # Time per spike includes the refractory period
    total_time = t_spike + tau_r
    return 1 / total_time

# Compute firing rates
f_no_refractory = firing_rate_no_refractory(I_e_values, tau, V_threshold)
f_with_refractory = firing_rate_with_refractory(I_e_values, tau, V_threshold, tau_r)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(I_e_values, f_no_refractory, label='Without Refractoriness')
plt.plot(I_e_values, f_with_refractory, label='With Refractoriness (tau_r = 2 ms)', linestyle='--')
plt.title("Firing Rate vs Input Current (f-I curve)")
plt.xlabel("Input Current I_e (arbitrary units)")
plt.ylabel("Firing Rate (spikes per ms)")
plt.legend()
plt.grid(True)
plt.savefig("integrate-and-fire-results/refractory-vs-non-refractory")
plt.show()
