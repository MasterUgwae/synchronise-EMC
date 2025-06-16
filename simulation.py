# simulation.py

import numpy as np

def kuramoto_derivative(theta, omega, K):
    """
    Compute the derivative dθ/dt for the Kuramoto model.
    
    Parameters:
      theta: numpy array of oscillator phases.
      omega: array of natural frequencies.
      K: coupling strength.
    
    Returns:
      A numpy array representing the time derivative for each oscillator.
    """
    N = theta.size
    # Use broadcasting to compute pairwise phase differences:
    # diff[i, j] = theta[j] - theta[i]
    diff = theta[None, :] - theta[:, None]
    # For each oscillator i, sum sin(theta[j] - theta[i]) for j from 1 to N.
    interaction = np.sum(np.sin(diff), axis=1)
    return omega + (K / N) * interaction

def simulate_kuramoto(theta0, omega, K, t_eval):
    """
    Simulate the Kuramoto model using an Euler integration method.
    
    Parameters:
      theta0: Initial phases (numpy array).
      omega: Natural frequencies (numpy array).
      K: Coupling strength.
      t_eval: Array of time values for simulation.
    
    Returns:
      A tuple (t_eval, theta_sol) where theta_sol is a 2D array holding phase values 
      for all oscillators at each time step.
    """
    dt = t_eval[1] - t_eval[0]
    N = theta0.size
    theta_sol = np.zeros((len(t_eval), N))
    theta_sol[0] = theta0.copy()
    
    for i in range(1, len(t_eval)):
        dtheta = kuramoto_derivative(theta_sol[i - 1], omega, K)
        theta_sol[i] = theta_sol[i - 1] + dt * dtheta
        # Keep phases within [0, 2π)
        theta_sol[i] = np.mod(theta_sol[i], 2 * np.pi)
        
    return t_eval, theta_sol

def compute_order_parameter(theta):
    """
    Calculate the Kuramoto order parameter for a given phase configuration.
    
    Parameters:
      theta: numpy array of oscillator phases at a given time.
    
    Returns:
      The order parameter r, defined as r = |(1/N) Σ exp(i θ)|.
      This quantifies the synchronisation level (r=1 implies full synchronisation).
    """
    return np.abs(np.mean(np.exp(1j * theta)))

# Optional: You can use the following block to run a demo simulation.
if __name__ == '__main__':
    import config
    import matplotlib.pyplot as plt

    # Run the simulation using configuration parameters.
    t, theta_sol = simulate_kuramoto(config.theta0, config.omega, config.K, config.t_eval)
    
    # Compute final order parameter.
    r_final = compute_order_parameter(theta_sol[-1])
    print(f"Final synchronisation order parameter: r = {r_final:.8f}")
    if r_final > 0.9:

        print("The oscillators are highly synchronised.")
    
        for i in range(config.N):
            print(f'The phase of oscillator {i+1} is {theta_sol[i]}')
    
    # Plot phase evolution for each oscillator.
    for i in range(config.N):
        plt.plot(t, theta_sol[:, i], label=f'Oscillator {i + 1}')
    plt.xlabel("Time")
    plt.ylabel("Phase (radians)")
    plt.title("Kuramoto Model Simulation (Euler integration)")
    plt.legend()
    plt.show()
