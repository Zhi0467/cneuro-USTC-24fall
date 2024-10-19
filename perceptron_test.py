from perceptron_model import *
import numpy as np
from tabulate import tabulate
print("-----------------------------------------------------------")
print("This program is used to test the perceptron model.")
print("it takes a long time to run (about 5 minutes), so please be patient.")
print("the solution will be shown sequentially in the console / terminal.")
print("the maximum number of epochs is set to 1000, the number of runs is set to 50, and learning rate is set to 0.1.")
print("-----------------------------------------------------------")
print("")

N_values = [20, 50, 100]
P_values = [20, 50, 100]
runs = 50
print("-----------------------------------------------------------")
print("solution for problem 1.1 -- collect stats")
stats = collect_stats(N_values, P_values, runs)
# show the stats 
for key, value in stats.items():
    print(f"features N={key[0]}, samples P={key[1]}: converged percentage={value['converged_percentage']}, average epochs={value['average_epochs']}")
    print("")
print("-----------------------------------------------------------")
print("")

# Compare convergence speed for P = 10 and N = 10, 20, 100
N_values_comparison = [10, 20, 100]
P_comparison = 10
print("-----------------------------------------------------------")
print("solution for problem 1.2 -- compare convergence speed at a fixed number of samples P=10")
convergence_stats = compare_convergence_speed(N_values_comparison, P_comparison)

# Display the results
print("\nConvergence speed comparison:")
headers = ["N", "Converged (%)", "Avg. Epochs"]
table_data = [[N, f"{stats['converged_percentage']:.2f}%", f"{stats['average_epochs']:.2f}"] 
              for N, stats in convergence_stats.items()]
print(tabulate(table_data, headers=headers, tablefmt="grid"))
print("-----------------------------------------------------------")
print("")

# Plot convergence probability for N=100 and alpha range [0, 3]
N = 128
alpha_range = (0, 3)
print("-----------------------------------------------------------")
print("solution for problem 1.3 -- plot convergence probability for N=128 and alpha range [0, 3], figure saved in the folder perceptron-results")
alpha_values, convergence_probs = plot_convergence_probability(N, alpha_range)
# Print some key points
print("\nKey points:")
# Find where probability drops below 50%
for alpha, prob in zip(alpha_values[::-1], convergence_probs[::-1]):
    if prob > 0.99:
        print(f"Convergence probability drops below 99% at α ≈ {alpha:.2f}")
        break
    if prob >= 0.5:
        print(f"Convergence probability drops below 50% after α ≈ {alpha:.2f}")
        break
print("-----------------------------------------------------------")
print("")

# Plot margin histogram for N=100 and alpha range [0, 3]
N = 128
alpha_range = (0, 3)
print("-----------------------------------------------------------")
print("solution for problem 1.4 -- plot margin histogram for N=128 and alpha range [0, 3], figure saved in the folder perceptron-results")
alpha_values, mean_margins = plot_margin_histogram(N, alpha_range)
print("\nMargin Analysis:")
print(f"Maximum mean margin: {max(mean_margins):.4f} at α ≈ {alpha_values[np.argmax(mean_margins)]:.2f}")

print("\nEffect of P and N on hyperplane margin:")
max_margin_index = np.argmax(mean_margins)
if max_margin_index < len(mean_margins) // 2:
    print("- The maximum mean margin occurs at a relatively low α value.")
    print("- This suggests that for a fixed N, smaller P (fewer samples) tends to result in larger margins.")
    print("- However, very small P may lead to poor generalization despite large margins.")
else:
    print("- The maximum mean margin occurs at a relatively high α value.")
    print("- This suggests that for a fixed N, larger P (more samples) tends to result in larger margins.")
    print("- This could indicate better generalization as more data points are correctly separated.")
print("-----------------------------------------------------------")
