import numpy as np
import matplotlib.pyplot as plt


# Constants

kB = 1.380649e-23          # J/K
T  = 1.0e4                 # K
mH = 1.6735575e-27         # kg
eV = 1.602176634e-19       # J

dE = 10.2 * eV             # J (Hydrogen n=1 -> 2 excitation energy)


# Maxwell–Boltzmann speed distribution f(v)

def f_MB(v):
    A = 4.0*np.pi * (mH/(2.0*np.pi*kB*T))**1.5
    return A * v**2 * np.exp(-mH*v**2/(2.0*kB*T))


# 3) RK4 numerical integration for an integral:
#    F = ∫ f_MB(v) dv from a to b

def rk4_integral(f, a, b, dv):
    N = int(np.ceil((b - a)/dv))
    v = a
    I = 0.0

    for _ in range(N):
        # RK4 "slopes"
        k1 = f(v)
        k2 = f(v + 0.5*dv)
        k3 = f(v + 0.5*dv)
        k4 = f(v + dv)

        I += (dv/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        v += dv

    return I


# (a) Plot f(v)

v_plot = np.linspace(0, 80000, 2000)
plt.figure()
plt.plot(v_plot, f_MB(v_plot))
plt.xlabel("v (m/s)")
plt.ylabel("f(v)")
plt.title("Maxwell–Boltzmann speed distribution (H, T=10,000 K)")
plt.tight_layout()
plt.savefig("p2a_MB_plot.png", dpi=200)

# (b) Compute vmin and fraction F

vmin = np.sqrt(2.0*dE/mH)
print("vmin =", vmin, "m/s")

# approximate infinity with a large vmax
dv   = 50.0
vmax = 3.0e5

F = rk4_integral(f_MB, vmin, vmax, dv)
print("Fraction F (RK4) =", F)

# (c) Convergence checks

# (c1) Change dv (step-size convergence) keeping vmax fixed
print("\nStep-size convergence (fixed vmax = 3e5):")
for dv_test in [200.0, 100.0, 50.0, 25.0]:
    F_test = rk4_integral(f_MB, vmin, vmax, dv_test)
    print("dv =", dv_test, " -> F =", F_test)

# (c2) Change vmax (upper-limit convergence) keeping dv fixed

print("\nUpper-limit convergence (fixed dv = 50):")
dv_fixed = 50.0
for vmax_test in [8.0e4, 1.0e5, 1.5e5, 2.0e5, 3.0e5]:
    F_test = rk4_integral(f_MB, vmin, vmax_test, dv_fixed)
    print("vmax =", vmax_test, " -> F =", F_test)

plt.show()