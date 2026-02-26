import numpy as np

def f(x, y):
    return y**2 + 1.0

def euler(x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(x[n], y[n])
    return x, y

def rk2_midpoint(x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + 0.5*h, y[n] + 0.5*h*k1)
        y[n+1] = y[n] + h * k2
    return x, y

def rk4(x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for n in range(N):
        k1 = f(x[n], y[n])
        k2 = f(x[n] + 0.5*h, y[n] + 0.5*h*k1)
        k3 = f(x[n] + 0.5*h, y[n] + 0.5*h*k2)
        k4 = f(x[n] + h, y[n] + h*k3)
        y[n+1] = y[n] + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y


x0, y0 = 0.0, 0.0
x_end = 3.0
N = 50

xE, yE = euler(x0, y0, x_end, N)
xR, yR = rk4(x0, y0, x_end, N)
#xR, yR = rk2_midpoint(x0, y0, x_end, N)
y_exact = np.tan(xE)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(xE, y_exact, marker=".", label="Exact tan(x)")
plt.plot(xE, yE, marker="*", label="Euler")
plt.plot(xR, yR, marker="x", color="red", label="RK4")
#plt.plot(xR, yR, label="RK2")
plt.ylim(-3, 50)  
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.savefig("Euler_RK.png", dpi=200)
plt.legend()
plt.tight_layout()
plt.show()