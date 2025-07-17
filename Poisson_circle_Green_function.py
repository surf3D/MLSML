import numpy as np

def greens_function(r, theta, rp, thetap):
    # Vectorized Green's function for the unit disk, Dirichlet BC
    delta_theta = theta - thetap
    A = r**2 + rp**2 - 2 * r * rp * np.cos(delta_theta)
    B = 1 - 2 * r * rp * np.cos(delta_theta) + (r * rp)**2
    G = -1/(4 * np.pi) * (np.log(A) - np.log(B))
    return G

def poisson_solution_on_disk(source_func, r_eval, theta_eval, Nr=50, Ntheta=100):
    # Discretize the source domain
    r_src = np.linspace(0, 1, Nr)
    theta_src = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
    dr = r_src[1] - r_src[0]
    dtheta = theta_src[1] - theta_src[0]

    # Meshgrid for integration points
    R_src, Theta_src = np.meshgrid(r_src, theta_src, indexing='ij')

    # Evaluate source at grid points
    S = source_func(R_src, Theta_src)

    # Prepare output
    u = np.zeros_like(r_eval, dtype=np.float64)

    # Compute solution at each (r_eval, theta_eval)
    for idx, (r0, theta0) in enumerate(zip(r_eval, theta_eval)):
        # Compute Green's function for all source points
        G = greens_function(r0, theta0, R_src, Theta_src)
        # Integrate: sum over all source points
        integrand = G * S * R_src  # R_src is the Jacobian for polar coordinates
        u[idx] = np.sum(integrand) * dr * dtheta

    return u

# Example usage:
if __name__ == "__main__":
    # Example source: s(r, theta) = 1 everywhere
    def source(r, theta):
        s = np.ones_like(r)
        s = np.sin(theta)
        return s

    # Evaluation points (e.g., 10 points along r=0.5, theta in [0, 2pi))
    N_eval = 10
    r_eval = np.full(N_eval, 0.5)
    theta_eval = np.linspace(0, 2*np.pi, N_eval, endpoint=False)

    u = poisson_solution_on_disk(source, r_eval, theta_eval)

    print("u(r=0.5, theta):")
    for th, val in zip(theta_eval, u):
        print(f"theta={th:.2f}, u={val:.6f}")
