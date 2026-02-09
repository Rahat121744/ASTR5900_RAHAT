import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

# Parameters
m = 1.0
k = 1.0
omega = np.sqrt(k/m)
t = np.linspace(0, 10, 20)          # discrete measurements
x = np.sin(omega * t)               # true SHO positions

# Linear interpolation
lin_interp = interp1d(t, x)
t_fine = np.linspace(0, 10, 500)
x_lin = lin_interp(t_fine)

# Cubic spline
cs_interp = CubicSpline(t, x)
x_cs = cs_interp(t_fine)

# Compute approximate energies
v_lin = np.gradient(x_lin, t_fine)
v_cs = np.gradient(x_cs, t_fine)
KE_lin = 0.5*m*v_lin**2
PE_lin = 0.5*k*x_lin**2
E_lin = KE_lin + PE_lin

KE_cs = 0.5*m*v_cs**2
PE_cs = 0.5*k*x_cs**2
E_cs = KE_cs + PE_cs

plt.plot(t_fine, E_lin, label="Linear Energy")
plt.plot(t_fine, E_cs, label="Cubic Spline Energy")
plt.axhline(0.5*k, color='k', linestyle='--', label="True Total Energy")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Mechanical Energy")
plt.show()
