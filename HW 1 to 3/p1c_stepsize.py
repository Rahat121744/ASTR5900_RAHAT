import numpy as np

def f(x,y):
    return y**2 + 1

def euler(x0,y0,x_end,N):
    h=(x_end-x0)/N
    x=np.linspace(x0,x_end,N+1)
    y=np.zeros(N+1); y[0]=y0
    for n in range(N):
        y[n+1]=y[n]+h*f(x[n],y[n])
    return y[-1]

def rk4(x0,y0,x_end,N):
    h=(x_end-x0)/N
    x=np.linspace(x0,x_end,N+1)
    y=np.zeros(N+1); y[0]=y0
    for n in range(N):
        k1=f(x[n],y[n])
        k2=f(x[n]+h/2,y[n]+h*k1/2)
        k3=f(x[n]+h/2,y[n]+h*k2/2)
        k4=f(x[n]+h,y[n]+h*k3)
        y[n+1]=y[n]+h*(k1+2*k2+2*k3+k4)/6
    return y[-1]

x_end=1.3
exact=np.tan(x_end)

Ns=[20,40,80,160,320]

print("N   Euler_err    RK4_err")
for N in Ns:
    e=euler(0,0,x_end,N)
    r=rk4(0,0,x_end,N)
    print(N, abs(e-exact), abs(r-exact))