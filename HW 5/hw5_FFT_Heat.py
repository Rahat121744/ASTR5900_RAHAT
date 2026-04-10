import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import time
import os

# ============================================================
# Create folder for figures
# ============================================================
os.makedirs("figures", exist_ok=True)

# ============================================================
# Function definition
# ============================================================
def f(x):
    return np.exp(-50.0 * (x - 0.5)**2)

# ============================================================
# Manual DFT
# ============================================================
def dft(xn):
    N = len(xn)
    Xk = np.zeros(N, dtype=complex)
    for k in range(N):
        total = 0.0 + 0.0j
        for n in range(N):
            total += xn[n] * np.exp(-2j * np.pi * k * n / N)
        Xk[k] = total
    return Xk

# ============================================================
# Manual IDFT
# ============================================================
def idft(Xk):
    N = len(Xk)
    xn = np.zeros(N, dtype=complex)
    for n in range(N):
        total = 0.0 + 0.0j
        for k in range(N):
            total += Xk[k] * np.exp(2j * np.pi * k * n / N)
        xn[n] = total / N
    return xn

# ============================================================
# Analytical Fourier Transform from Q1(a)
# ============================================================
def analytic_F(k):
    return np.sqrt(np.pi / 50.0) * np.exp(-k**2 / 200.0) * np.exp(-1j * k / 2.0)

# ============================================================
# Question 1(b): DFT vs analytical transform
# Also verify inverse transform
# ============================================================
N_values = [32, 64, 128]

plt.figure(figsize=(8, 5))

for N in N_values:
    dx = 1.0 / N
    x = np.arange(N) * dx
    fx = f(x)

    X = dft(fx)
    F_num = X / N
    k = 2.0 * np.pi * np.arange(N)

    plt.plot(k[:20], np.abs(F_num[:20]), 'o', label=f'N={N}')

k_plot = 2.0 * np.pi * np.arange(20)
plt.plot(k_plot, np.abs(analytic_F(k_plot)), 'k-', linewidth=2, label='Analytical')

plt.xlabel('k')
plt.ylabel(r'$|F(k)|$')
plt.title('Q1(b): DFT vs Analytical Fourier Transform')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q1b_dft_vs_analytical.png", dpi=300, bbox_inches='tight')
plt.close()

# Inverse transform reconstruction example for N = 64
N = 64
dx = 1.0 / N
x = np.arange(N) * dx
fx = f(x)

X = dft(fx)
fx_rec = idft(X)

plt.figure(figsize=(8, 5))
plt.plot(x, fx, 'k-', linewidth=2, label='Original')
plt.plot(x, fx_rec.real, 'ro', markersize=4, label='Reconstructed')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Q1(b): Original vs Reconstructed Function')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q1b_reconstruction.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Question 1(c): FFT vs DFT comparison
# ============================================================
N = 128
dx = 1.0 / N
x = np.arange(N) * dx
fx = f(x)

X_dft = dft(fx)
X_fft = fft(fx)

F_dft = X_dft / N
F_fft = X_fft / N
k = 2.0 * np.pi * np.arange(N)

plt.figure(figsize=(8, 5))
plt.plot(k[:20], np.abs(F_dft[:20]), 'ro', label='DFT')
plt.plot(k[:20], np.abs(F_fft[:20]), 'bx', label='FFT')
plt.xlabel('k')
plt.ylabel(r'$|F(k)|$')
plt.title('Q1(c): DFT vs FFT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q1c_dft_vs_fft.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Question 1(c): Timing study
# DFT: N = 2^j, j = 3,...,11
# FFT: N = 2^j, j = 3,...,20
# Average over 100 runs
# ============================================================
def time_dft(N, runs=100):
    x = np.linspace(0, 1, N, endpoint=False)
    fx = f(x)
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        dft(fx)
        total += time.perf_counter() - start
    return total / runs

def time_fft(N, runs=100):
    x = np.linspace(0, 1, N, endpoint=False)
    fx = f(x)
    total = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        fft(fx)
        total += time.perf_counter() - start
    return total / runs

N_dft = [2**j for j in range(3, 12)]
N_fft = [2**j for j in range(3, 21)]

t_dft = []
t_fft = []

print("Timing DFT...")
for N in N_dft:
    td = time_dft(N, runs=100)
    t_dft.append(td)
    print(f"DFT  N={N:6d}, avg time = {td:.6e} s")

print("Timing FFT...")
for N in N_fft:
    tf = time_fft(N, runs=100)
    t_fft.append(tf)
    print(f"FFT  N={N:6d}, avg time = {tf:.6e} s")

plt.figure(figsize=(8, 5))
plt.loglog(N_dft, t_dft, 'ro-', label='DFT')
plt.loglog(N_fft, t_fft, 'bx-', label='FFT')
plt.xlabel('N')
plt.ylabel('Execution time (s)')
plt.title('Q1(c): Time Complexity of DFT vs FFT')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig("figures/q1c_time_complexity.png", dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Question 2(a): Solve heat equation in Fourier space
# ============================================================
N = 64
L = 1.0
dx = L / N
x = np.arange(N) * dx

alpha = 0.005
dt = 0.001
T = 5.0
steps = int(T / dt)

u = f(x)
u_hat = fft(u)
k = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

U = np.zeros((steps + 1, N))
heat = np.zeros(steps + 1)
times = np.linspace(0, T, steps + 1)

U[0] = u
heat[0] = dx * np.sum(u)

for i in range(1, steps + 1):
    u_hat = u_hat - alpha * k**2 * u_hat * dt
    u = np.real(ifft(u_hat))
    U[i] = u
    heat[i] = dx * np.sum(u)

plt.figure(figsize=(8, 5))
plt.imshow(U, aspect='auto', origin='lower', extent=[0, 1, 0, T])
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Q2(a): Heat Equation Evolution')
plt.tight_layout()
plt.savefig("figures/q2a_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(times, heat, 'k-')
plt.xlabel('t')
plt.ylabel('Total Heat')
plt.title('Q2(a): Total Heat vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q2a_heat_conservation.png", dpi=300, bbox_inches='tight')
plt.close()

# Plot a few time snapshots
snapshot_indices = [0, steps//10, steps//2, steps]
snapshot_labels = [f"t={times[idx]:.2f}" for idx in snapshot_indices]

plt.figure(figsize=(8, 5))
for idx, label in zip(snapshot_indices, snapshot_labels):
    plt.plot(x, U[idx], label=label)
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Q2(a): Solution Snapshots')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q2a_snapshots.png", dpi=300, bbox_inches='tight')
plt.close()

# =========================
# Q2(b): Instability study
# =========================
alpha2 = 0.05
dt2 = 0.01
T2 = 5.0
steps2 = int(T2 / dt2)

u = f(x)
u_hat = fft(u)

U2 = []
heat2 = []
t2 = []

blow_step = None

for i in range(steps2):
    u_hat = u_hat - alpha2 * k**2 * u_hat * dt2
    u = np.real(ifft(u_hat))

    if not np.all(np.isfinite(u)):
        blow_step = i
        print(f"Instability detected at step {i}, t = {i*dt2:.3f}")
        break

    U2.append(u.copy())
    heat2.append(dx * np.sum(u))
    t2.append((i + 1) * dt2)

U2 = np.array(U2)
heat2 = np.array(heat2)
t2 = np.array(t2)

# heatmap of unstable evolution
plt.figure(figsize=(8, 5))
plt.imshow(U2, aspect='auto', origin='lower',
           extent=[0, 1, t2[0] if len(t2)>0 else 0, t2[-1] if len(t2)>0 else 0])
plt.colorbar(label='u(x,t)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Q2(b): Unstable Heat Evolution')
plt.tight_layout()
plt.savefig("figures/q2b_instability.png", dpi=300, bbox_inches='tight')
plt.close()

# last finite snapshot
if len(U2) > 0:
    last_idx = len(U2) - 1
    plt.figure(figsize=(8, 5))
    plt.plot(x, U2[last_idx], 'r-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Q2(b): Last Finite Snapshot at t = {t2[last_idx]:.3f}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/q2b_instability_snapshot.png", dpi=300, bbox_inches='tight')
    plt.close()

# total heat
plt.figure(figsize=(8, 5))
plt.plot(t2, heat2, 'k-')
plt.xlabel('t')
plt.ylabel('Total Heat')
plt.title('Q2(b): Total Heat vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/q2b_heat_conservation.png", dpi=300, bbox_inches='tight')
plt.close()