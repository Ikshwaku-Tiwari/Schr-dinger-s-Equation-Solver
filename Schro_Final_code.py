import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define spatial grid
x_min = -50
x_max = 50
N = 2000
dx = (x_max - x_min) / (N - 1)
x = np.linspace(x_min, x_max, N)

# Define time grid
T = 10
dt = 0.01
n_steps = int(T / dt)
t = np.linspace(0, T, n_steps)

# Define potential function
def V(x):
    return 0.5 * x**2


# Define initial wavefunction
def psi_init(x):
    return np.exp(-0.5 * (x - 0)**2)

psi = psi_init(x).astype(complex)

# Define Hamiltonian operator with periodic boundary conditions
def H(psi, x, V, dx):
    psi_fft = np.fft.fft(psi)
    k = np.fft.fftfreq(len(x), dx / (2 * np.pi))
    psi_fft *= np.exp(-1j * 0.5 * k**2 * dt)
    psi_fft *= np.exp(-1j * V(x) * dt)
    psi_new = np.fft.ifft(psi_fft)
    return psi_new



# Define animation function with periodic boundary conditions
def animate(i):
    psi_new = H(psi, x, V, dx)
    psi_new /= np.sqrt(np.sum(np.abs(psi_new)**2) * dx)
    psi_new[0] = psi_new[-2] # Set periodic boundary condition at x_min
    psi_new[-1] = psi_new[1] # Set periodic boundary condition at x_max
    line.set_ydata(np.abs(psi_new)**2)
    psi[:] = psi_new
    return line,


# Set up plot
fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('|psi(x,t)|^2')
ax.set_facecolor('black')
line, = ax.plot(x, np.abs(psi)**2, c='red')

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=n_steps, interval=30, blit=True)

# Show animation
plt.show()
