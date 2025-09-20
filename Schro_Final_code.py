import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


x_min = -50
x_max = 50
N = 2000
dx = (x_max - x_min) / (N - 1)
x = np.linspace(x_min, x_max, N)

T = 10
dt = 0.01
n_steps = int(T / dt)
t = np.linspace(0, T, n_steps)


def V(x):
    return 0.5 * x**2


def psi_init(x):
    return np.exp(-0.5 * (x - 0)**2)


psi = psi_init(x).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi)**2) * dx)



def H(psi, x, V, dx, dt):
    
    
    psi_new = psi * np.exp(-1j * V(x) * dt / 2.0)
    psi_fft = np.fft.fft(psi_new)
    k = np.fft.fftfreq(len(x), dx / (2 * np.pi)) 
    psi_fft *= np.exp(-1j * 0.5 * k**2 * dt)
    psi_new = np.fft.ifft(psi_fft)
    psi_new *= np.exp(-1j * V(x) * dt / 2.0)
    
    return psi_new



def animate(i):
    psi_new = H(psi, x, V, dx, dt)
    psi_new /= np.sqrt(np.sum(np.abs(psi_new)**2) * dx)
    line.set_ydata(np.abs(psi_new)**2)
    psi[:] = psi_new 
    
    return line,


fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.5)
ax.set_xlabel('x (Position)')
ax.set_ylabel('$|\psi(x,t)|^2$ (Probability Density)')
ax.set_title('Quantum Harmonic Oscillator Simulation')
ax.set_facecolor('black')
line, = ax.plot(x, np.abs(psi)**2, c='red', lw=2)


ani = animation.FuncAnimation(fig, 
                              animate, 
                              frames=n_steps, 
                              interval=30, 
                              blit=True)

plt.show()
