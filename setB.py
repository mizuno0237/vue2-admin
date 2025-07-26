import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt

# Supply Chain Problem Data
# Three suppliers with different characteristics
cost = np.array([10, 15, 20])  # Economic: cost per unit (minimize)
emission = np.array([5, 3, 8])  # Environmental: emission per unit (minimize)
employment = np.array([30, 20, 40])  # Social: employment generated per unit (maximize)
demand = 100  # Total demand constraint

print("=== SUSTAINABLE SUPPLY CHAIN MULTI-OBJECTIVE OPTIMIZATION ===")
print(f"Supplier data:")
print(f"Cost per unit: {cost}")
print(f"Emission per unit: {emission}")
print(f"Employment per unit: {employment}")
print(f"Total demand: {demand}")

# Calculate actual achievable ranges
print(f"\nAchievable ranges:")
print(f"Cost: [{np.min(cost) * demand:.0f} - {np.max(cost) * demand:.0f}]")
print(f"Emission: [{np.min(emission) * demand:.0f} - {np.max(emission) * demand:.0f}]")
print(f"Employment: [{np.min(employment) * demand:.0f} - {np.max(employment) * demand:.0f}]")
print()

# Constraint: total supply must meet demand
A_eq = np.array([[1, 1, 1]])  # x1 + x2 + x3 = demand
b_eq = np.array([demand])
bounds = [(0, None)] * 3  # Non-negativity constraints

results = {}
solutions = {}

# Method 1: Aggregated Approach (Simple Sum)
print("1. AGGREGATED METHOD (Simple Sum)")
print("-" * 40)
# Simple aggregation without normalization first to see base case
c_aggregated = cost + emission - employment  # Simple sum
res_aggregated = linprog(c_aggregated, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res_aggregated.success:
    agg_cost = np.dot(cost, res_aggregated.x)
    agg_emission = np.dot(emission, res_aggregated.x)
    agg_employment = np.dot(employment, res_aggregated.x)
    results['Aggregated'] = [agg_cost, agg_emission, agg_employment]
    solutions['Aggregated'] = res_aggregated.x
    print(f"Optimal solution: {res_aggregated.x.round(2)}")
    print(f"Cost: {agg_cost:.2f}, Emission: {agg_emission:.2f}, Employment: {agg_employment:.2f}")
else:
    print("Aggregated method failed")

# Method 2: Weighted Sum Method
print("\n2. WEIGHTED SUM METHOD")
print("-" * 40)
w_cost, w_emission, w_employment = 0.3, 0.3, 0.4  # Balanced weights

# Scale objectives to similar magnitude before weighting
cost_scale = 1000  # cost baseline
emission_scale = 500  # emission baseline
employment_scale = 3000  # employment baseline

c_weighted = (w_cost * cost / cost_scale * 1000 +
              w_emission * emission / emission_scale * 1000 -
              w_employment * employment / employment_scale * 1000)

res_weighted = linprog(c_weighted, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res_weighted.success:
    weighted_cost = np.dot(cost, res_weighted.x)
    weighted_emission = np.dot(emission, res_weighted.x)
    weighted_employment = np.dot(employment, res_weighted.x)
    results['Weighted Sum'] = [weighted_cost, weighted_emission, weighted_employment]
    solutions['Weighted Sum'] = res_weighted.x
    print(f"Weights: Cost={w_cost}, Emission={w_emission}, Employment={w_employment}")
    print(f"Optimal solution: {res_weighted.x.round(2)}")
    print(f"Cost: {weighted_cost:.2f}, Emission: {weighted_emission:.2f}, Employment: {weighted_employment:.2f}")
else:
    print("Weighted sum method failed")

# Method 3: Epsilon-Constraint Method
print("\n3. EPSILON-CONSTRAINT METHOD")
print("-" * 40)
emission_limit = 450  # More relaxed emission constraint
print(f"Minimizing cost with emission constraint ≤ {emission_limit}")
# Minimize cost subject to emission constraint
A_ub = np.array([emission])  # emission constraint
b_ub = np.array([emission_limit])  # emission upper bound
res_epsilon = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if res_epsilon.success:
    eps_cost = np.dot(cost, res_epsilon.x)
    eps_emission = np.dot(emission, res_epsilon.x)
    eps_employment = np.dot(employment, res_epsilon.x)
    results['Epsilon-Constraint'] = [eps_cost, eps_emission, eps_employment]
    solutions['Epsilon-Constraint'] = res_epsilon.x
    print(f"Optimal solution: {res_epsilon.x.round(2)}")
    print(f"Cost: {eps_cost:.2f}, Emission: {eps_emission:.2f}, Employment: {eps_employment:.2f}")
else:
    print("Epsilon-constraint method failed")

# Method 4: Lexicographic Method
print("\n4. LEXICOGRAPHIC METHOD")
print("-" * 40)
print("Priority order: 1) Cost, 2) Emission, 3) Employment")
# Step 1: Minimize cost
res_lex1 = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
if res_lex1.success:
    optimal_cost = np.dot(cost, res_lex1.x)
    print(f"Step 1 - Optimal cost: {optimal_cost:.2f}")

    # Step 2: Among solutions with optimal cost, minimize emission
    # Allow small tolerance for numerical stability
    cost_tolerance = optimal_cost * 1.001
    A_ub_lex = np.array([cost])
    b_ub_lex = np.array([cost_tolerance])
    res_lex2 = linprog(emission, A_ub=A_ub_lex, b_ub=b_ub_lex, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res_lex2.success:
        optimal_emission = np.dot(emission, res_lex2.x)
        print(f"Step 2 - Optimal emission: {optimal_emission:.2f}")

        # Step 3: Among solutions with optimal cost and emission, maximize employment
        emission_tolerance = optimal_emission + 0.01
        A_ub_lex3 = np.vstack([cost, emission])
        b_ub_lex3 = np.array([cost_tolerance, emission_tolerance])
        res_lex3 = linprog(-employment, A_ub=A_ub_lex3, b_ub=b_ub_lex3, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                           method='highs')

        if res_lex3.success:
            lex_cost = np.dot(cost, res_lex3.x)
            lex_emission = np.dot(emission, res_lex3.x)
            lex_employment = np.dot(employment, res_lex3.x)
            results['Lexicographic'] = [lex_cost, lex_emission, lex_employment]
            solutions['Lexicographic'] = res_lex3.x
            print(f"Final solution: {res_lex3.x.round(2)}")
            print(f"Cost: {lex_cost:.2f}, Emission: {lex_emission:.2f}, Employment: {lex_employment:.2f}")
        else:
            # Use step 2 result if step 3 fails
            results['Lexicographic'] = [np.dot(cost, res_lex2.x), optimal_emission, np.dot(employment, res_lex2.x)]
            solutions['Lexicographic'] = res_lex2.x
            print(f"Using step 2 result: {res_lex2.x.round(2)}")
    else:
        # Use step 1 result if step 2 fails
        results['Lexicographic'] = [optimal_cost, np.dot(emission, res_lex1.x), np.dot(employment, res_lex1.x)]
        solutions['Lexicographic'] = res_lex1.x
        print(f"Using step 1 result: {res_lex1.x.round(2)}")
else:
    print("Lexicographic method failed")

# Method 5: Goal Programming
print("\n5. GOAL PROGRAMMING")
print("-" * 40)
# Set achievable targets
target_cost = 1200  # Mid-range target
target_emission = 450  # Mid-range target
target_employment = 3200  # Challenging but achievable target
print(f"Targets: Cost={target_cost}, Emission={target_emission}, Employment={target_employment}")


def goal_programming_objective(x):
    # Ensure all values are non-negative
    x = np.maximum(x, 0)

    actual_cost = np.dot(cost, x)
    actual_emission = np.dot(emission, x)
    actual_employment = np.dot(employment, x)

    # Calculate relative deviations
    dev_cost = abs(actual_cost - target_cost) / target_cost
    dev_emission = abs(actual_emission - target_emission) / target_emission
    dev_employment = abs(actual_employment - target_employment) / target_employment

    return dev_cost + dev_emission + dev_employment


# Constraints
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x) - demand},  # demand constraint
]

# Bounds to ensure non-negative solutions
goal_bounds = [(0, demand) for _ in range(3)]

x0 = np.array([30, 30, 40])  # Better initial guess
res_goal = minimize(goal_programming_objective, x0, method='SLSQP',
                    constraints=constraints, bounds=goal_bounds,
                    options={'ftol': 1e-9, 'disp': False})

if res_goal.success:
    # Clean solution
    x_solution = np.maximum(res_goal.x, 0)
    # Ensure constraint satisfaction
    x_solution = x_solution * demand / np.sum(x_solution)

    goal_cost = np.dot(cost, x_solution)
    goal_emission = np.dot(emission, x_solution)
    goal_employment = np.dot(employment, x_solution)
    results['Goal Programming'] = [goal_cost, goal_emission, goal_employment]
    solutions['Goal Programming'] = x_solution
    print(f"Optimal solution: {x_solution.round(2)}")
    print(f"Cost: {goal_cost:.2f}, Emission: {goal_emission:.2f}, Employment: {goal_employment:.2f}")
    print(f"Total weighted deviation: {res_goal.fun:.4f}")
else:
    print("Goal programming failed")

# Results Summary and Visualization
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

if results:
    # Create comparison table
    print(f"{'Method':<20} {'Cost':<10} {'Emission':<12} {'Employment':<12}")
    print("-" * 60)
    for method, values in results.items():
        print(f"{method:<20} {values[0]:<10.1f} {values[1]:<12.1f} {values[2]:<12.1f}")

    # Create clean visualization
    methods = list(results.keys())
    values = np.array(list(results.values()))

    # Create figure with proper sizing
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Create grouped bar chart
    x = np.arange(3)
    width = 0.15

    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values[i], width, label=method,
                      color=colors[i % len(colors)], alpha=0.8)

        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                    f'{height:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Customize the plot
    ax.set_ylabel('Value', fontsize=14, fontweight='bold')
    ax.set_title('Comparison of Objectives under Multi-objective Optimization Methods',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Cost', 'Emission', 'Employment'], fontsize=12, fontweight='bold')

    # Position legend outside plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis to start from 0 and adjust upper limit
    ax.set_ylim(0, np.max(values) * 1.1)

    plt.tight_layout()
    plt.show()

    # Enhanced Analysis
    print(f"\nDETAILED ANALYSIS:")
    print("-" * 40)

    # Find best method for each objective
    costs = [v[0] for v in results.values()]
    emissions = [v[1] for v in results.values()]
    employments = [v[2] for v in results.values()]

    best_cost_idx = np.argmin(costs)
    best_emission_idx = np.argmin(emissions)
    best_employment_idx = np.argmax(employments)

    print(f"Best for Cost Minimization: {methods[best_cost_idx]} (Cost: {costs[best_cost_idx]:.1f})")
    print(f"Best for Emission Reduction: {methods[best_emission_idx]} (Emission: {emissions[best_emission_idx]:.1f})")
    print(
        f"Best for Employment Creation: {methods[best_employment_idx]} (Employment: {employments[best_employment_idx]:.1f})")

    # Calculate trade-offs
    cost_range = max(costs) - min(costs)
    emission_range = max(emissions) - min(emissions)
    employment_range = max(employments) - min(employments)

    print(f"\nTrade-off Analysis:")
    print(f"Cost variation across methods: {cost_range:.1f} ({cost_range / min(costs) * 100:.1f}%)")
    print(f"Emission variation: {emission_range:.1f} ({emission_range / min(emissions) * 100:.1f}%)")
    print(f"Employment variation: {employment_range:.1f} ({employment_range / max(employments) * 100:.1f}%)")

    print(f"\nKEY INSIGHTS:")
    print("• Methods show meaningful differences in objective trade-offs")
    print("• Cost-focused methods (Aggregated, Lexicographic) prioritize economic efficiency")
    print("• Constraint-based methods (ε-Constraint) effectively manage environmental limits")
    print("• Goal Programming balances all three sustainability dimensions")
    print("• Weighted Sum allows flexible stakeholder preference incorporation")

    print(f"\nRECOMMENDED APPROACH:")
    print("Goal Programming is recommended for this sustainable supply chain problem because:")
    print("- Balances economic, environmental, and social objectives simultaneously")
    print("- Allows explicit target setting aligned with sustainability frameworks")
    print("- Provides interpretable deviation measures for performance tracking")
    print("- Supports stakeholder engagement in target definition")
    print("- Enables adaptive management through target adjustment")
    print("- Facilitates ESG reporting and regulatory compliance")

else:
    print("No methods solved successfully - check problem formulation")