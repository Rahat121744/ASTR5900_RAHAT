import numpy as np
import matplotlib.pyplot as plt

# Kepler equation
def f(E, e, M):
    return E - e*np.sin(E) - M

def df(E, e):
    return 1 - e*np.cos(E)

# Newton–Raphson with convergence history
def newton_kepler(M, e, E0=0.5, tol=1e-8, max_iter=100):
    E = E0
    residuals = []

    print("Iter    E                |f(E)|")

    for i in range(max_iter):
        FE = f(E, e, M)
        residuals.append(abs(FE))

        print(i, E, abs(FE))

        if abs(FE) < tol:
            return E, i+1, residuals

        E = E - FE/df(E, e)

    raise RuntimeError("Did not converge")

# Parameters
M = 1.0        # mean anomaly (radians)
e = 0.5        # eccentricity

# Run solver
root, steps, res = newton_kepler(M, e)

print("\nRoot E =", root)
print("Iterations =", steps)

# -----------------------
# Convergence plot
# -----------------------
plt.figure()
plt.semilogy(range(len(res)), res, marker='o')
plt.xlabel("Iteration")
plt.ylabel("|f(E)|")
plt.title("Newton–Raphson Convergence for Kepler's Equation (e = 0.5)")
plt.tight_layout()

plt.savefig("convergence_kepler.png")
plt.show()
