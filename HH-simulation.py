# this file is a solution to HW3-2
# we simulate the HH model under a specific set of parameter values
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
gK_max = 36  # Maximum conductance of K+ (mS/cm^2)
gNa_max = 120  # Maximum conductance of Na+ (mS/cm^2)
gL = 0.3     # Leak conductance (mS/cm^2)
EK = -77     # Reversal potential for K+ (mV)
ENa = 50     # Reversal potential for Na+ (mV)
EL = -54.387 # Leak reversal potential (mV)
Cm = 1       # Membrane capacitance (µF/cm^2)

# Gating variable rate functions
def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55)))
def beta_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40)))
def beta_m(V): return 4 * np.exp(-0.0556 * (V + 65))
def alpha_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def beta_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))

# Define the system of ODEs
def hodgkin_huxley(y, t, I_e):
    V, n, m, h = y

    # Currents
    INa = gNa_max * (m**3) * h * (V - ENa)
    IK = gK_max * (n**4) * (V - EK)
    IL = gL * (V - EL)

    # Membrane potential differential
    dVdt = (I_e - (IK + INa + IL)) / Cm

    # Gating variables differentials
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h

    return [dVdt, dndt, dmdt, dhdt]

# Initial conditions
V0 = -65   
n0 = alpha_n(V0) / (alpha_n(V0) + beta_n(V0))
m0 = alpha_m(V0) / (alpha_m(V0) + beta_m(V0))
h0 = alpha_h(V0) / (alpha_h(V0) + beta_h(V0))
y0 = [V0, n0, m0, h0]

# Time vector
t = np.linspace(0, 120, 20000)  # Time in ms

# External current (constant stimulus)
I_e = [6, 6.15, 6.25, 7, 10] # Applied current (nA/mm^2)
V = np.array([]) 
spike_frequencies = []

# Function to detect spikes
def detect_spikes(V, t):
    V = np.array(V)
    threshold = 0  # Set a threshold for detecting spikes
    spike_times = t[np.where((V[1:] >= threshold) & (V[:-1] < threshold))]
    return spike_times

# Solve ODEs
plt.figure(figsize=(10, 6))
for i in I_e:
    solution = odeint(hodgkin_huxley, y0, t, args=(i,))
    # Extract variables
    V = solution[:, 0]

    # Detect spikes
    spike_times = detect_spikes(V, t)
    
    # Calculate spike frequency (Hz)
    if len(spike_times) > 2:
        spike_intervals = np.diff(spike_times)  # Time intervals between spikes
        frequency = 1000 / np.mean(spike_intervals)  # Frequency in Hz
    else:
        frequency = 0  # No spikes
    
    spike_frequencies.append(frequency)
    plt.plot(t, V, label=f"I_e = {i}")

filename = f"HH-simulation-results/Hodgkin-Huxley Neuron Action Potential"
plt.title(f"Hodgkin-Huxley Neuron Action Potential")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.legend()
plt.grid()
plt.savefig(filename)
plt.show()

# Plot frequency as a function of current
plt.figure()
plt.plot(I_e, spike_frequencies, marker='o')
plt.title('Spike Frequency as a Function of Input Current')
plt.xlabel('Input Current (µA/cm^2)')
plt.ylabel('Spike Frequency (Hz)')
plt.grid()
plt.savefig("HH-simulation-results/frequency-and-external-current")
plt.show()
