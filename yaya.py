import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt

# Example data
cost = np.array([10, 15, 20])         # Economic objective: cost
emission = np.array([5, 3, 8])        # Environmental objective: emission
employment = np.array([30, 20, 40])   # Social objective: employment (maximize, so use -employment for minimization)
demand = 100                          # Total demand

# Constraint matrix
A = np.array([[1, 1, 1]])
b = np.array([demand])
bounds = [(0, None)] * 3

results = {}

print("=== Weighted Sum Method ===")
w_cost, w_emission, w_employment = 0.4, 0.3, 0.3
c_weighted = w_cost * cost + w_emission * emission - w_employment * employment
res_weighted = linprog(c_weighted, A_ub=-A, b_ub=-b, bounds=bounds)
if not res_weighted.success:
    print("Weighted sum method failed:", res_weighted.message)
else:
    weighted_vals = [np.dot(cost, res_weighted.x), np.dot(emission, res_weighted.x), np.dot(employment, res_weighted.x)]
    results['Weighted Sum'] = weighted_vals
    print(f"Optimal solution: {res_weighted.x.round(2)}")
    print(f"Cost={weighted_vals[0]:.2f}  Emission={weighted_vals[1]:.2f}  Employment={weighted_vals[2]:.2f}")

print("\n=== Epsilon-Constraint Method (Cost as main objective, Emission upper bound) ===")
A_epsilon = np.vstack([A, emission])
b_epsilon = np.append(b, 400)
res_epsilon = linprog(cost, A_ub=-A_epsilon, b_ub=-b_epsilon, bounds=bounds)
if not res_epsilon.success:
    print("Epsilon-constraint method failed:", res_epsilon.message)
else:
    epsilon_vals = [np.dot(cost, res_epsilon.x), np.dot(emission, res_epsilon.x), np.dot(employment, res_epsilon.x)]
    results['Epsilon-Constraint'] = epsilon_vals
    print(f"Optimal solution: {res_epsilon.x.round(2)}")
    print(f"Cost={epsilon_vals[0]:.2f}  Emission={epsilon_vals[1]:.2f}  Employment={epsilon_vals[2]:.2f}")

print("\n=== Goal Programming (Targets: 1200, 400, 3500) ===")
def goal_programming(x):
    target_cost = 1200
    target_emission = 400
    target_employment = 3500
    d_cost = abs(np.dot(cost, x) - target_cost)
    d_emission = abs(np.dot(emission, x) - target_emission)
    d_employment = abs(np.dot(employment, x) - target_employment)
    return d_cost + d_emission + d_employment

cons = ({'type': 'ineq', 'fun': lambda x: np.sum(x) - demand},
        {'type': 'ineq', 'fun': lambda x: x})
x0 = np.array([demand/3]*3)
res_goal = minimize(goal_programming, x0, constraints=cons)
if not res_goal.success:
    print("Goal programming failed:", res_goal.message)
else:
    goal_vals = [np.dot(cost, res_goal.x), np.dot(emission, res_goal.x), np.dot(employment, res_goal.x)]
    results['Goal Programming'] = goal_vals
    print(f"Optimal solution: {res_goal.x.round(2)}")
    print(f"Cost={goal_vals[0]:.2f}  Emission={goal_vals[1]:.2f}  Employment={goal_vals[2]:.2f}")

# Visualization output
if results:
    labels = ['Cost', 'Emission', 'Employment']
    methods = list(results.keys())
    vals = np.array(list(results.values()))

    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        ax.bar(x + i*width, vals[i], width, label=method)

    ax.set_ylabel('Value')
    ax.set_title('Comparison of Objectives under Three Multi-objective Optimization Methods')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()

    for i in range(len(methods)):
        for j in range(len(labels)):
            ax.text(x[j] + i*width, vals[i, j] + max(vals[:, j])*0.01, f'{vals[i, j]:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
else:
    print("No method solved successfully, cannot plot comparison chart.")
