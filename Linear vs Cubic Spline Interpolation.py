import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Load the data
file_path = r'C:\Users\Admin\Desktop\Git\ASTR5900_RAHAT\HW01_data.txt'
# Inspecting the file format first
df = pd.read_csv(file_path, sep='\t')
print(df.head())
print(df.info())

# Extract x and y
x_data = df['x'].values
y_data = df['y'].values

# 2. Linear Interpolation Function
def linear_interpolate(x_eval, x_data, y_data):
    """
    Linearly interpolates y values for given x_eval based on x_data and y_data.
    Handles extrapolation by using constant slope from the nearest segment.
    """
    y_eval = []
    for x in x_eval:
        # Find the interval [x_i, x_{i+1}] containing x
        if x < x_data[0]:
            # Extrapolation (below min) - use first segment slope
            slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
            y = y_data[0] + slope * (x - x_data[0])
        elif x > x_data[-1]:
            # Extrapolation (above max) - use last segment slope
            slope = (y_data[-1] - y_data[-2]) / (x_data[-1] - x_data[-2])
            y = y_data[-1] + slope * (x - x_data[-1])
        else:
            # Interpolation
            # Find index i such that x_data[i] <= x <= x_data[i+1]
            # Using searchsorted to find the index
            i = np.searchsorted(x_data, x)
            if i > 0 and i < len(x_data):
                 # x is between index i-1 and i
                 # Adjust index to be left side of interval
                 idx = i - 1
            elif i == 0:
                 idx = 0 # Should match x_data[0]
            else:
                 idx = len(x_data) - 2 # Should match x_data[-1]

            x0, x1 = x_data[idx], x_data[idx+1]
            y0, y1 = y_data[idx], y_data[idx+1]
            
            slope = (y1 - y0) / (x1 - x0)
            y = y0 + slope * (x - x0)
        y_eval.append(y)
    return np.array(y_eval)

# Generate higher resolution x (10 times more points)
# Number of intervals = len(x_data) - 1
# Original points N. New points ~ 10 * N
n_interp = len(x_data) * 10
x_fine = np.linspace(x_data.min(), x_data.max(), n_interp)

# Apply Linear Interpolation
y_linear = linear_interpolate(x_fine, x_data, y_data)

# 2b. Off-the-shelf Cubic Spline
cs = CubicSpline(x_data, y_data)
y_cubic = cs(x_fine)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ko', label='Original Data', markersize=8, zorder=5)
plt.plot(x_fine, y_linear, 'b--', label='Linear Interpolation (Custom)', linewidth=2)
plt.plot(x_fine, y_cubic, 'r-', label='Cubic Spline (Scipy)', linewidth=2, alpha=0.7)

plt.title('Linear vs Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('interpolation_plot.png')