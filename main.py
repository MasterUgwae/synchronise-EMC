# main.py
import numpy as np
import cmath
from scipy.integrate import solve_ivp
from scipy.stats import cauchy
import matplotlib.pyplot as plt

# Define the function for the Kuramoto model ODE
def kuramoto(t, theta, omega, K):
    N = len(theta)
    dtheta_dt = np.zeros(N)
    # Calculate the derivative for each oscillator
    for i in range(N):
        # Sum the sine of the differences between oscillator i and all others
        interaction = np.sum(np.sin(theta - theta[i]))
        dtheta_dt[i] = omega[i] + (K / N) * interaction
    return dtheta_dt

# Parameters for the model
N = 10             # Number of oscillators
gamma = 1
x_0 = 50
omega = cauchy.rvs(loc = x_0, scale = 1, size=N)
K = 3.0                            # Coupling strength
theta0 = np.random.uniform(0, 2 * np.pi, N)  # Initial phases randomly in [0, 2π]

# Time span for the simulation
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 3000)  # Evaluation points

def kuramoto_rotating(t, phi, omega, K, x0):
    N = len(phi)
    dphi_dt = np.zeros(N)
    for i in range(N):
        interaction = np.sum(np.sin(phi - phi[i]))
        # omega minus the baseline rotation x0
        dphi_dt[i] = (omega[i] - x0) + (K / N) * interaction
    return dphi_dt

x0 = 50  # mean frequency, matching your cauchy.rvs distribution location parameter
solution = solve_ivp(
    lambda t, y: kuramoto_rotating(t, y, omega, K, x0),
    t_span, theta0, t_eval=t_eval, method='RK45'
)
total=cmath.rect(0,0)
# Plot the results
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.plot(solution.t, solution.y[i, :], label=f'Oscillator {i+1}')
    total+=cmath.rect(1,solution.y[i, :][-1])

total/=N

r, phi = cmath.polar(total)

print(f"The polar form of {total} is {r}e^(i{phi})")

plt.xlabel('Time')
plt.ylabel('Phase, θ')
plt.title('Kuramoto Model Simulation using solve_ivp')
plt.legend()
plt.show()
