import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return y**2 + 1.0

def euler(x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1); y[0] = y0
    for n in range(N):
        y[n+1] = y[n] + h * f(x[n], y[n])
    return x, y

def rk4(x0, y0, x_end, N):
    h = (x_end - x0) / N
    x = np.linspace(x0, x_end, N+1)
    y = np.zeros(N+1); y[0] = y0
    for n in range(N):
        xn, yn = x[n], y[n]
        k1 = f(xn, yn)
        k2 = f(xn + 0.5*h, yn + 0.5*h*k1)
        k3 = f(xn + 0.5*h, yn + 0.5*h*k2)
        k4 = f(xn + h,     yn + h*k3)
        y[n+1] = yn + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return x, y

# ---- settings for part (b) ----
x0, y0 = 0.0, 0.0
x_end = 3.0
N = 50          

x, yE = euler(x0, y0, x_end, N)
x, yR = rk4(x0, y0, x_end, N)
yExact = np.tan(x)

errE = np.abs(yE - yExact)
errR = np.abs(yR - yExact)

# ---- Plot : errors ----
plt.figure()
plt.plot(x, errE, label="Euler |error|")
plt.plot(x, errR, label="RK4 |error|")
plt.ylim(-1, 4)
plt.xlabel("x")
plt.ylabel("absolute error")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("p1b_errors.png", dpi=200)
plt.show()