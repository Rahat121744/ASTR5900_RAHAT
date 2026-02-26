import numpy as np
import matplotlib.pyplot as plt

#The ODE: dy/dx = y^2 + 1
def f(x, y):
    return y**2 + 1.0

def solve(method, x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        if method == 'euler':
            y[n+1] = y[n] + h * f(x[n], y[n]) # 
        elif method == 'rk4':
            k1 = h * f(x[n], y[n])
            k2 = h * f(x[n] + h/2, y[n] + k1/2)
            k3 = h * f(x[n] + h/2, y[n] + k2/2)
            k4 = h * f(x[n] + h, y[n] + k3)
            y[n+1] = y[n] + (k1 + 2*k2 + 2*k3 + k4) / 6 
    return x, y

Ns = [50, 100, 200, 500, 1000, 2000]
x_end = 1.2
hs, err_euler, err_rk4 = [], [], []

# Better value from finest RK4
_, y_fine = solve('rk4', 0, 0, x_end, 5000)
y_true = y_fine[-1]

#Convergence Study 
for N in Ns:
    h = x_end / N
    hs.append(h)
    
    _, yE = solve('euler', 0, 0, x_end, N)
    _, yR = solve('rk4', 0, 0, x_end, N)
    
    err_euler.append(abs(yE[-1] - y_true) / y_true)
    err_rk4.append(abs(yR[-1] - y_true) / y_true)

# Plotting Results 
plt.figure()
plt.loglog(hs, err_euler, '-o', label='Euler')
plt.loglog(hs, err_rk4, '-s', label='RK4')
plt.xlabel('Step Size $h$')
plt.ylabel('Fractional Error')
plt.title('Convergence Study: Euler vs RK4')
plt.grid(True)
plt.savefig('p1d_convergence.png')
plt.legend()
plt.show()

# Calculation for 1(e)
y_max = yR[-1]
y_exact = np.tan(1.2)

delta_conv = abs(err_rk4[-1] - err_rk4[-2]) 
delta_true = abs(y_max - y_exact) / abs(y_exact)

print(f"Convergence Uncertainty (Delta Conv): {delta_conv}")
print(f"True Fractional Error (Delta True):   {delta_true}")