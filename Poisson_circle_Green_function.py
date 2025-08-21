import numpy as np

def greens_function(r, theta, rp, thetap):
    delta_theta = theta - thetap
    A = r**2 + rp**2 - 2 * r * rp * np.cos(delta_theta)
    B = 1 - 2 * r * rp * np.cos(delta_theta) + (r * rp)**2
    G = -1/(4 * np.pi) * (np.log(A) - np.log(B))
    return G

def poisson_solution_on_disk_nodes(source_func, nodes, Nr=50, Ntheta=100):
    # nodes: shape (N, 2), columns are x and y
    x_eval = nodes[:, 0]
    y_eval = nodes[:, 1]

    # Discretize the source domain in polar coordinates
    r_src = np.linspace(0, 1, Nr)
    theta_src = np.linspace(0, 2*np.pi, Ntheta, endpoint=False)
    dr = r_src[1] - r_src[0]
    dtheta = theta_src[1] - theta_src[0]

    # Meshgrid for integration points
    R_src, Theta_src = np.meshgrid(r_src, theta_src, indexing='ij')
    X_src = R_src * np.cos(Theta_src)
    Y_src = R_src * np.sin(Theta_src)

    # Evaluate source at grid points (in x, y)
    S = source_func(X_src, Y_src)

    # Prepare output
    u = np.zeros(len(nodes), dtype=np.float64)

    # Convert evaluation points to polar coordinates
    r_eval = np.sqrt(x_eval**2 + y_eval**2)
    theta_eval = np.arctan2(y_eval, x_eval)

    # Compute solution at each (x_eval, y_eval)
    for idx, (r0, theta0) in enumerate(zip(r_eval, theta_eval)):
        G = greens_function(r0, theta0, R_src, Theta_src)
        integrand = G * S * R_src  # Jacobian
        u[idx] = np.sum(integrand) * dr * dtheta

    return u
 

