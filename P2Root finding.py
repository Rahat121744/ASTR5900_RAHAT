import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
def f(x):
    return x**3 - 7*x**2 + 14*x - 5

def df(x):
    return 3*x**2 - 14*x + 14

# 1. Bisection Method
def bisection_method(func, a, b, tol):
    if func(a) * func(b) >= 0:
        return None, 0, []  # Bisection fails if signs are same

    steps = 0
    history = []
    
    while (b - a) / 2.0 > tol:
        steps += 1
        midpoint = (a + b) / 2.0
        history.append(midpoint)
        
        if func(midpoint) == 0:
            return midpoint, steps, history
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
            
    return (a + b) / 2.0, steps, history

# 2. Newton-Raphson Method
def newton_raphson(func, dfunc, x0, tol, max_iter=100):
    x = x0
    steps = 0
    history = [x]
    
    for i in range(max_iter):
        steps += 1
        fx = func(x)
        dfx = dfunc(x)
        
        if dfx == 0:
            print("Derivative is zero. No solution found.")
            return None, steps, history
            
        x_new = x - fx / dfx
        history.append(x_new)
        
        if abs(x_new - x) < tol:
            return x_new, steps, history
            
        x = x_new
        
    return x, steps, history

# --- Execute for Problem 2a ---
tolerance = 1e-8

# Bisection (0, 1)
root_bi, steps_bi, hist_bi = bisection_method(f, 0, 1, tolerance)
print(f"Bisection Result: {root_bi:.9f} in {steps_bi} steps")

# Newton-Raphson (x0=0)
root_nr, steps_nr, hist_nr = newton_raphson(f, df, 0, tolerance)
print(f"Newton-Raphson Result: {root_nr:.9f} in {steps_nr} steps")

# --- Plotting ---
x_vals = np.linspace(-1, 5, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='$f(x) = x^3 - 7x^2 + 14x - 5$')
plt.axhline(0, color='black', lw=1)
plt.axvline(root_nr, color='r', linestyle='--', label=f'Root $\\approx {root_nr:.4f}$')

# Plot critical points (where derivative is zero) for discussion 2c
crit_1 = (14 - np.sqrt(14**2 - 4*3*14))/(6) # Quadratic formula for 3x^2 - 14x + 14
crit_2 = (14 + np.sqrt(14**2 - 4*3*14))/(6)
plt.plot([crit_1, crit_2], [f(crit_1), f(crit_2)], 'ro', label='Local Extrema')

plt.title('Root Finding Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('P2Root finding_plot.png')