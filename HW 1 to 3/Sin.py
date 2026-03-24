import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Define the true function
def true_function(x):
    return np.sin((np.pi / 2) * x) + (x / 2)

# 2. Create dataset (integers 0 to 10)
x_data = np.arange(0, 11, 1) # 0 to 10 inclusive
y_data = true_function(x_data)

# 3. High resolution grid (10x resolution)
# Interval 0 to 10. Original points = 11.
# 10x resolution -> 10 * 10 intervals = 100 intervals? Or just 10x points.
# Let's just use linspace with plenty of points for smooth plotting, e.g., 100 or 101 points.
# "interpolate the data to at least 10 times higher resolution"
x_fine = np.linspace(0, 10, 101) 
y_true_fine = true_function(x_fine)

# 4. Linear Interpolation (Custom function from before)
def linear_interpolate(x_eval, x_data, y_data):
    y_eval = []
    for x in x_eval:
        # We know x is within [0, 10] for this problem, but let's keep logic robust
        if x < x_data[0]:
            slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
            y = y_data[0] + slope * (x - x_data[0])
        elif x > x_data[-1]:
            slope = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])
            y = y_data[-1] + slope * (x - x_data[-1])
        else:
            idx = np.searchsorted(x_data, x) - 1
            idx = max(0, min(idx, len(x_data) - 2))
            x0, x1 = x_data[idx], x_data[idx+1]
            y0, y1 = y_data[idx], y_data[idx+1]
            slope = (y1 - y0) / (x1 - x0)
            y = y0 + slope * (x - x0)
        y_eval.append(y)
    return np.array(y_eval)

y_linear_fine = linear_interpolate(x_fine, x_data, y_data)

# 5. Cubic Spline Interpolation
cs = CubicSpline(x_data, y_data)
y_cubic_fine = cs(x_fine)

# 6. Relative Error
# Avoid division by zero at x=0 where y_true=0.
# We will mask the value at x=0 or use a very small epsilon.
# Since x_fine[0] is exactly 0, we can just mask it.
with np.errstate(divide='ignore', invalid='ignore'):
    rel_error_linear = (y_linear_fine - y_true_fine) / y_true_fine
    rel_error_cubic = (y_cubic_fine - y_true_fine) / y_true_fine

# Replace infinity/NaN at x=0 with 0 or drop it for plotting
# At x=0, both interpolations should be exact (it's a knot), so error is 0/0.
# The limit is technically defined, but computationally messy. 
# We'll just set the error at x=0 to 0 because the interpolation passes through the point.
rel_error_linear[0] = 0.0
rel_error_cubic[0] = 0.0


# Plot 1: Function Comparison
plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(x_fine, y_true_fine, 'k-', label='True Function', linewidth=1.5, alpha=0.5)
plt.plot(x_data, y_data, 'ko', label='Sampled Data', zorder=5)
plt.plot(x_fine, y_linear_fine, 'b--', label='Linear Interp', linewidth=1.5)
plt.plot(x_fine, y_cubic_fine, 'r:', label='Cubic Spline', linewidth=2)
plt.title('Q3a: Interpolation vs True Function')
plt.legend()
plt.grid(True)
plt.ylabel('y')

# Plot 2: Relative Error
plt.subplot(2, 1, 2)
plt.plot(x_fine, rel_error_linear, 'b--', label='Linear Relative Error')
plt.plot(x_fine, rel_error_cubic, 'r-', label='Cubic Spline Relative Error')
plt.title('Q3b: Relative Error [(Interp - True) / True]')
plt.xlabel('x')
plt.ylabel('Relative Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('q3_plots.png')

print("Linear Max Rel Error:", np.max(np.abs(rel_error_linear)))
print("Cubic Max Rel Error:", np.max(np.abs(rel_error_cubic)))