import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import solve
from scipy.sparse import coo_matrix
import optimize_driver 

def mls_poisson_solution(all_nodes, boundary_nodes, interior_nodes, optimize_driver, weight_function, ds):
    N_total = all_nodes.shape[0]
    N_boundary = boundary_nodes.shape[0]
    N_interior = interior_nodes.shape[0]
    tree = cKDTree(all_nodes)
    neighbor_radius = 2 * ds
    neighbors = tree.query_ball_point(interior_nodes, neighbor_radius)

    def p(x, y):
        return np.array([1, x, y, x**2, x*y, y**2])

    def weight(x, y, xj, yj, h=2*ds):
        r2 = (x - xj)**2 + (y - yj)**2
        return np.exp(-r2 / h**2)

    def laplacian_weight(x, y, xj, yj, h=2*ds):
        r2 = (x - xj)**2 + (y - yj)**2
        w = np.exp(-r2 / h**2)
        return ((4 * r2) / h**4 - 4 / h**2) * w

    def source_func(x, y):
        return 1.0

    lap_p = np.array([0, 0, 0, 2, 0, 2])
    A_data = []
    A_row = []
    A_col = []
    b = np.zeros(N_interior)

    for i, (xi, yi) in enumerate(interior_nodes):
        idx_neighbors = neighbors[i]
        xj = all_nodes[idx_neighbors, 0]
        yj = all_nodes[idx_neighbors, 1]
        pj = np.stack([p(x, y) for x, y in zip(xj, yj)], axis=1)
        Wj = np.array([weight(xi, yi, x, y) for x, y in zip(xj, yj)])
        A_mat = (pj * Wj) @ pj.T
        A_inv = np.linalg.pinv(A_mat)
        for j, (xj_, yj_) in enumerate(zip(xj, yj)):
            pj_ = p(xj_, yj_)
            term1 = lap_p.T @ A_inv @ (pj_ * Wj[j])
            lap_w = laplacian_weight(xi, yi, xj_, yj_)
            term2 = pj_.T @ A_inv @ p(xi, yi) * lap_w
            coeff = term1 + term2
            A_row.append(i)
            A_col.append(idx_neighbors[j])
            A_data.append(coeff)
        b[i] = source_func(xi, yi)

    A = coo_matrix((A_data, (A_row, A_col)), shape=(N_interior, N_total)).tocsr()
    A_interior = A[:, N_boundary:]
    u_interior = solve(A_interior.toarray(), b)

    u = np.zeros(N_total)
    u[N_boundary:] = u_interior
    return u
